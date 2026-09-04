#!/usr/bin/env python3
"""Run GR00T-N1.6 static inference on one Bimanual YAM episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path("/coc/testnvme/xzhang3205/vla-adaptation")
GR00T_ROOT = REPO_ROOT / "models/gr00t-n1.6"
STATIC_ROOT = REPO_ROOT / "static-inference"
YAM_ROOT = STATIC_ROOT / "molmoact-yam"
sys.path[:0] = [str(YAM_ROOT), str(STATIC_ROOT), str(REPO_ROOT), str(GR00T_ROOT)]

from contracts import right_arm_gripper
from data import PartialYAMDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

CHECKPOINT = REPO_ROOT / "checkpoints/gr00t/gr00t-n1.6"
EMBODIMENT = EmbodimentTag.ROBOCASA_PANDA_OMRON
HORIZON = 16
NUM_STEPS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument("--no-gradnorm", action="store_true")
    return parser.parse_args()


def load_neutral_values(checkpoint: Path) -> tuple[dict, dict]:
    payload = json.loads((checkpoint / "statistics.json").read_text())
    statistics = payload["robocasa_panda_omron"]
    states = {key: np.asarray(value["mean"], dtype=np.float32) for key, value in statistics["state"].items()}
    actions = {key: np.asarray(value["mean"], dtype=np.float32) for key, value in statistics["action"].items()}
    return states, actions


def build_inputs(policy, content):
    processed = policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    collated = _rec_to_dtype(policy.collate_fn([processed]), dtype=torch.bfloat16)
    mask = collated["inputs"]["action_mask"]
    time_mask = mask.any(dim=-1, keepdim=True)
    mask.zero_()
    mask[..., :7] = time_mask
    return collated


def main():
    args = parse_args()
    policy = Gr00tPolicy(embodiment_tag=EMBODIMENT, model_path=args.checkpoint, device=args.device)
    neutral_states, neutral_actions = load_neutral_values(args.checkpoint)
    dataset = PartialYAMDataset(args.dataset_root)
    trajectory = dataset[args.trajectory_index]
    frame_count = max(0, len(trajectory) - HORIZON + 1)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    losses = [[] for _ in range(NUM_STEPS)]
    cosines = [[] for _ in range(NUM_STEPS)]
    gradnorms = [[] for _ in range(NUM_STEPS)]
    targets, velocities = [], [[] for _ in range(NUM_STEPS)]

    for frame in range(frame_count):
        state = right_arm_gripper(trajectory.state[frame])
        actions = right_arm_gripper(trajectory.actions[frame : frame + HORIZON])
        rotation = neutral_states["end_effector_rotation_relative"].copy()
        rotation[:3] = state[3:6]
        gripper = neutral_states["gripper_qpos"].copy()
        gripper[0] = state[6]
        content = VLAStepData(
            images={
                "res256_image_side_0": [trajectory.image("observation.images.top", frame)],
                "res256_image_side_1": [trajectory.image("observation.images.left", frame)],
                "res256_image_wrist_0": [trajectory.image("observation.images.right", frame)],
            },
            masks=None,
            states={
                "end_effector_position_relative": state[:3][None],
                "end_effector_rotation_relative": rotation[None],
                "gripper_qpos": gripper[None],
                "base_position": neutral_states["base_position"][None],
                "base_rotation": neutral_states["base_rotation"][None],
            },
            actions={
                "end_effector_position": actions[:, :3],
                "end_effector_rotation": actions[:, 3:6],
                "gripper_close": actions[:, 6:7],
                "base_motion": np.broadcast_to(neutral_actions["base_motion"], (HORIZON, 4)).copy(),
                "control_mode": np.broadcast_to(neutral_actions["control_mode"], (HORIZON, 1)).copy(),
            },
            text=trajectory.instruction,
            embodiment=EMBODIMENT,
        )
        result = policy.model.static_inference(
            **build_inputs(policy, content), compute_gradnorm=not args.no_gradnorm
        )
        for step in range(NUM_STEPS):
            losses[step].append(result["final_loss"][step])
            cosines[step].append(result["cosine"][step])
            gradnorms[step].append(result["gradnorm_vision"][step])
        if args.save_meta:
            targets.append(result["u"][0, :HORIZON, :7].numpy())
            for step in range(NUM_STEPS):
                velocities[step].append(result["v"][step][0, :HORIZON, :7].numpy())

    for step in range(NUM_STEPS):
        np.save(args.output_dir / f"final_loss_{step}.npy", np.asarray(losses[step], dtype=np.float32))
        np.save(args.output_dir / f"cosine_{step}.npy", np.asarray(cosines[step], dtype=np.float32))
        np.save(args.output_dir / f"gradnorm_vision_step_{step}.npy", np.asarray(gradnorms[step], dtype=np.float32))
    if args.save_meta:
        meta = args.output_dir / "meta"
        meta.mkdir(exist_ok=True)
        np.save(meta / "u.npy", np.stack(targets).astype(np.float32))
        for step in range(NUM_STEPS):
            np.save(meta / f"v_{step}.npy", np.stack(velocities[step]).astype(np.float32))
    (args.output_dir / "trajectory_meta.json").write_text(
        json.dumps(
            {
                "model": "gr00t-n1.6",
                "embodiment": "robocasa_panda_omron",
                "episode_index": trajectory.episode_index,
                "source_length": len(trajectory),
                "action_horizon": HORIZON,
                "num_frames_used": frame_count,
                "side": "right",
                "metric_dims": list(range(7)),
                "instruction_source": "meta/tasks_annotated.parquet",
                "camera_mapping": {"base": "top", "primary_wrist": "right", "secondary_wrist": "left"},
            },
            indent=2,
        )
        + "\n"
    )
    dataset.clear_video_cache()


if __name__ == "__main__":
    main()

