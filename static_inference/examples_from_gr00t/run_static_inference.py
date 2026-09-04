"""Run static inference for one RoboCasa atomic seen task with GR00T-N1.6.

Static inference is analysis-only: for each frame of each demonstration episode,
the model runs its standard 4-step flow-matching denoising loop starting from a
single noise tensor, and the per-step velocity predictions are compared against
the ground-truth action chunk (u = gt_action - noise). Per-frame scalars are
stacked per episode and saved as npy files.

Output layout (see README.md for full semantics):

    <output_root>/<timestamp>/task_<i>/episode_<j:06d>/
        final_loss_{n}.npy             (T_ep,)  always saved, n in 0..3
        cosine_{n}.npy                 (T_ep,)  always saved
        gradnorm_vision_step_{n}.npy   (T_ep,)  always saved
        meta/u.npy                     (T_ep, 16, 12)  only with --save_meta
        meta/v_{n}.npy                 (T_ep, 16, 12)  only with --save_meta
    <output_root>/<timestamp>/task_<i>/summary.json

Frames per episode: t = 0 .. ep_len - 16 (full 16-step GT action chunk available,
matching training's valid steps, no end padding). Axis 0 of every file of an
episode is this frame index.

Example:
    python static_inference/run_static_inference.py --task_id 2 --save_meta
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
GR00T_ROOT = REPO_ROOT / "models" / "gr00t-n1.6"
sys.path.insert(0, str(GR00T_ROOT))

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

DEFAULT_DATASET_ROOT = REPO_ROOT / "datasets" / "robocasa365" / "atomic-seen-splits"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "gr00t" / "gr00t-n1.6"
DEFAULT_OUTPUT_ROOT = Path("/coc/testnvme/xzhang3205/static/gr00t")

EMBODIMENT_TAG = EmbodimentTag.ROBOCASA_PANDA_OMRON
NUM_DENOISE_STEPS = 4  # gr00t default num_inference_timesteps
ACTION_HORIZON = 16  # real action horizon (robocasa action delta indices 0..15)
ROBOCASA_ARM_GRIPPER_DIM = 7


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task_id", type=int, default=None, help="Task id, 1..18 (not needed with --dataset_dir)")
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Direct LeRobot task dir; overrides dataset_root/task_{task_id}_demo_50",
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--save_meta", action="store_true", help="Also save meta/u.npy and meta/v_{n}.npy")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_episodes", type=int, default=None, help="Debug: cap number of episodes")
    parser.add_argument("--max_frames", type=int, default=None, help="Debug: cap frames per episode")
    parser.add_argument("--num_steps", type=int, default=NUM_DENOISE_STEPS,
                        help="Number of denoising steps to compute (default: 4)")
    parser.add_argument("--timestamp", type=str, default=None, help="Override output timestamp dir name")
    parser.add_argument("--no_gradnorm", action="store_true", help="Skip vision grad norm computation")
    return parser.parse_args()


def build_frame_inputs(policy, data_point):
    """Mirror Gr00tPolicy._get_action's processor path (eval mode, no train-time
    augmentation), but keep the ground-truth action/action_mask from the processor."""
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": data_point}]
    processed = policy.processor(messages)
    collated = policy.collate_fn([processed])
    collated = _rec_to_dtype(collated, dtype=torch.bfloat16)
    collated["inputs"]["action_mask"][..., ROBOCASA_ARM_GRIPPER_DIM:] = 0
    return collated


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    NUM_DENOISE_STEPS = args.num_steps  # CLI override; shadows the module-level default (4)

    if args.dataset_dir is None:
        assert args.task_id is not None and 1 <= args.task_id <= 18, f"task_id must be in 1..18, got {args.task_id}"
        dataset_path = Path(args.dataset_root) / f"task_{args.task_id}_demo_50"
    else:
        dataset_path = Path(args.dataset_dir)
    assert dataset_path.is_dir(), f"Dataset not found: {dataset_path}"

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dataset_dir is not None and args.task_id is None:
        # --dataset_dir mode: output_root/timestamp already identifies the task.
        task_output_dir = Path(args.output_root) / timestamp
    else:
        task_output_dir = Path(args.output_root) / timestamp / f"task_{args.task_id}"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output dir: {task_output_dir}")

    policy = Gr00tPolicy(
        embodiment_tag=EMBODIMENT_TAG,
        model_path=args.checkpoint,
        device=args.device,
    )
    modality_configs = policy.get_modality_config()
    assert "action" in modality_configs, "Modality configs must include the action modality"
    logging.info(f"Modality config: \n{modality_configs}")

    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=modality_configs,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    num_episodes = len(loader)
    if args.max_episodes is not None:
        num_episodes = min(num_episodes, args.max_episodes)
    logging.info(f"Dataset: {dataset_path} ({len(loader)} episodes, running {num_episodes})")

    # Per-task aggregation over all frames of all episodes.
    task_final_loss = [[] for _ in range(NUM_DENOISE_STEPS)]
    task_cosine = [[] for _ in range(NUM_DENOISE_STEPS)]
    task_gradnorm = [[] for _ in range(NUM_DENOISE_STEPS)]
    total_frames = 0

    for ep_idx in range(num_episodes):
        traj = loader[ep_idx]
        ep_len = len(traj)
        # t = 0 .. ep_len - 16 inclusive: full GT action chunk available.
        frame_indices = list(range(0, ep_len - ACTION_HORIZON + 1))
        if args.max_frames is not None:
            frame_indices = frame_indices[: args.max_frames]
        if not frame_indices:
            logging.warning(f"Episode {ep_idx}: len {ep_len} < {ACTION_HORIZON}, skipping")
            continue

        ep_final_loss = [[] for _ in range(NUM_DENOISE_STEPS)]
        ep_cosine = [[] for _ in range(NUM_DENOISE_STEPS)]
        ep_gradnorm = [[] for _ in range(NUM_DENOISE_STEPS)]
        ep_u = []
        ep_v = [[] for _ in range(NUM_DENOISE_STEPS)]

        for t in frame_indices:
            data_point = extract_step_data(traj, t, modality_configs, EMBODIMENT_TAG)
            collated = build_frame_inputs(policy, data_point)
            result = policy.model.static_inference(
                **collated, num_inference_steps=args.num_steps, compute_gradnorm=not args.no_gradnorm
            )

            # Derive real action dims from the mask (horizon 16 x dim 12 for robocasa).
            mask = result["action_mask"][0]  # (max_action_horizon, max_action_dim), 0/1
            real_h = int(mask.any(dim=1).sum().item())
            real_d = int(mask.any(dim=0).sum().item())

            for n in range(NUM_DENOISE_STEPS):
                ep_final_loss[n].append(result["final_loss"][n])
                ep_cosine[n].append(result["cosine"][n])
                ep_gradnorm[n].append(result["gradnorm_vision"][n])
            if args.save_meta:
                ep_u.append(result["u"][0, :real_h, :real_d].numpy())
                for n in range(NUM_DENOISE_STEPS):
                    ep_v[n].append(result["v"][n][0, :real_h, :real_d].numpy())

            if t % 50 == 0:
                logging.info(
                    f"task {args.task_id} ep {ep_idx} frame {t}/{frame_indices[-1]} "
                    f"cos={['%.4f' % result['cosine'][n] for n in range(NUM_DENOISE_STEPS)]}"
                )

        ep_dir = task_output_dir / f"episode_{ep_idx:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        for n in range(NUM_DENOISE_STEPS):
            np.save(ep_dir / f"final_loss_{n}.npy", np.asarray(ep_final_loss[n], dtype=np.float32))
            np.save(ep_dir / f"cosine_{n}.npy", np.asarray(ep_cosine[n], dtype=np.float32))
            np.save(
                ep_dir / f"gradnorm_vision_step_{n}.npy",
                np.asarray(ep_gradnorm[n], dtype=np.float32),
            )
        if args.save_meta:
            meta_dir = ep_dir / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            np.save(meta_dir / "u.npy", np.stack(ep_u).astype(np.float32))
            for n in range(NUM_DENOISE_STEPS):
                np.save(meta_dir / f"v_{n}.npy", np.stack(ep_v[n]).astype(np.float32))

        for n in range(NUM_DENOISE_STEPS):
            task_final_loss[n].extend(ep_final_loss[n])
            task_cosine[n].extend(ep_cosine[n])
            task_gradnorm[n].extend(ep_gradnorm[n])
        total_frames += len(frame_indices)
        logging.info(
            f"Episode {ep_idx} done: {len(frame_indices)} frames, "
            f"mean cosine step0={np.mean(ep_cosine[0]):.4f}"
        )
        del traj

    summary = {
        "task_id": args.task_id,
        "dataset": str(dataset_path),
        "checkpoint": str(args.checkpoint),
        "timestamp": timestamp,
        "num_episodes": num_episodes,
        "total_frames": total_frames,
        "num_denoise_steps": NUM_DENOISE_STEPS,
        "save_meta": args.save_meta,
        "mean_final_loss": {str(n): float(np.mean(task_final_loss[n])) for n in range(NUM_DENOISE_STEPS)},
        "mean_cosine": {str(n): float(np.mean(task_cosine[n])) for n in range(NUM_DENOISE_STEPS)},
        "mean_gradnorm_vision": {
            str(n): float(np.nanmean(task_gradnorm[n])) for n in range(NUM_DENOISE_STEPS)
        },
    }
    with open(task_output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Task {args.task_id} done. Summary: {json.dumps(summary['mean_cosine'])}")


if __name__ == "__main__":
    main()
