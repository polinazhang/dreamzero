#!/usr/bin/env python3
"""Run reverse-mapped DROID static inference for GR00T-N1.6 on one trajectory."""

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
sys.path[:0] = [str(STATIC_ROOT), str(REPO_ROOT), str(GR00T_ROOT)]

from droid.archive import DroidTrajectory, manifest_entry
from droid.contracts import droid_cartesian_action, droid_state, reverse_map_action, reverse_map_state
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/gr00t/gr00t-n1.6"
EMBODIMENT = EmbodimentTag.ROBOCASA_PANDA_OMRON
HORIZON = 16
NUM_STEPS = 4
MODEL_ACTION_METRIC_DIMS = (0, 1, 2, 3, 4, 5, 6)


def model_action_metric_mask(width: int) -> np.ndarray:
    if width < 12:
        raise ValueError(f"mask width must be at least 12, got {width}")
    mask = np.zeros(width, dtype=np.float32)
    mask[list(MODEL_ACTION_METRIC_DIMS)] = 1.0
    return mask


def split_mapped_state(mapped_state: np.ndarray) -> dict[str, np.ndarray]:
    """Convert the raw 16-slot RoboCasa state layout into named processor modalities."""
    return {
        "end_effector_position_relative": mapped_state[7:10][None],
        "end_effector_rotation_relative": mapped_state[10:14][None],
        "gripper_qpos": mapped_state[14:16][None],
        "base_position": mapped_state[0:3][None],
        "base_rotation": mapped_state[3:7][None],
    }


def split_mapped_actions(mapped_actions: np.ndarray) -> dict[str, np.ndarray]:
    """Convert the raw 12-slot RoboCasa action layout into named processor modalities."""
    return {
        "end_effector_position": mapped_actions[:, 5:8],
        "end_effector_rotation": mapped_actions[:, 8:11],
        "gripper_close": mapped_actions[:, 11:12],
        "base_motion": mapped_actions[:, 0:4],
        "control_mode": mapped_actions[:, 4:5],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-meta", action="store_true")
    parser.add_argument("--no-gradnorm", action="store_true")
    return parser.parse_args()


def build_inputs(policy, content: VLAStepData):
    processed = policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
    collated = _rec_to_dtype(policy.collate_fn([processed]), dtype=torch.bfloat16)
    action_mask = collated["inputs"]["action_mask"]
    metric_mask = torch.as_tensor(
        model_action_metric_mask(action_mask.shape[-1]),
        device=action_mask.device,
    )
    collated["inputs"]["action_mask"] = action_mask * metric_mask
    return collated


def main():
    args = parse_args()
    entry = manifest_entry(args.manifest, args.trajectory_index)
    policy = Gr00tPolicy(embodiment_tag=EMBODIMENT, model_path=args.checkpoint, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    losses = [[] for _ in range(NUM_STEPS)]
    cosines = [[] for _ in range(NUM_STEPS)]
    gradnorms = [[] for _ in range(NUM_STEPS)]
    targets, velocities = [], [[] for _ in range(NUM_STEPS)]

    with DroidTrajectory(entry["archive_path"]) as trajectory:
        frame_count = max(0, len(trajectory) - HORIZON + 1)
        if args.max_frames is not None:
            frame_count = min(frame_count, args.max_frames)
        for frame in range(frame_count):
            native_state = droid_state(trajectory.arrays["observation_joint_position"][frame], trajectory.arrays["observation_gripper_position"][frame])
            mapped_state = reverse_map_state(native_state)
            native_actions = droid_cartesian_action(
                trajectory.arrays["action_cartesian_velocity"][frame : frame + HORIZON],
                trajectory.arrays["action_gripper_position"][frame : frame + HORIZON],
            )
            mapped_actions = reverse_map_action(native_actions)
            content = VLAStepData(
                images={
                    "res256_image_side_0": [trajectory.image("exterior_image_1_left", frame)],
                    "res256_image_side_1": [trajectory.image("exterior_image_2_left", frame)],
                    "res256_image_wrist_0": [trajectory.image("wrist_image_left", frame)],
                },
                masks=None,
                states=split_mapped_state(mapped_state),
                actions=split_mapped_actions(mapped_actions),
                text=trajectory.prompt,
                embodiment=EMBODIMENT,
            )
            result = policy.model.static_inference(**build_inputs(policy, content), compute_gradnorm=not args.no_gradnorm)
            for step in range(NUM_STEPS):
                losses[step].append(result["final_loss"][step])
                cosines[step].append(result["cosine"][step])
                gradnorms[step].append(result["gradnorm_vision"][step])
            if args.save_meta:
                targets.append(result["u"][0, :HORIZON, :12].numpy())
                for step in range(NUM_STEPS):
                    velocities[step].append(result["v"][step][0, :HORIZON, :12].numpy())

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
    (args.output_dir / "trajectory_meta.json").write_text(json.dumps({
        "model": "gr00t-n1.6",
        "embodiment": "robocasa_panda_omron",
        "trajectory_index": args.trajectory_index,
        "trajectory_id": entry["trajectory_id"],
        "source_length": entry["length"],
        "action_horizon": HORIZON,
        "num_frames_used": frame_count,
        "discarded_tail": min(HORIZON - 1, entry["length"]),
        "metric_dims_after_model_concatenation": list(MODEL_ACTION_METRIC_DIMS),
        "metric_dims_in_raw_robocasa_layout": [5, 6, 7, 8, 9, 10, 11],
        "action_source": "reverse-mapped cartesian_velocity + gripper_position",
        "language_field": "language_instruction",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
