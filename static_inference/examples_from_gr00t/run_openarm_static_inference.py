#!/usr/bin/env python3
"""Run GR00T-N1.6 static inference on one right-side OpenArm episode."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT=Path("/coc/testnvme/xzhang3205/vla-adaptation"); GR00T_ROOT=REPO_ROOT/"models/gr00t-n1.6"
sys.path[:0]=[str(REPO_ROOT/"static-inference"),str(REPO_ROOT),str(GR00T_ROOT)]
from openarm.contracts import right_action, right_state
from openarm.data import OpenArmTrajectory, manifest_entry
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

CHECKPOINT=REPO_ROOT/"checkpoints/gr00t/gr00t-n1.6"; HORIZON=16; NUM_STEPS=4; OVERFLOW_SLOT=12


def build_inputs(policy, content, joint7):
    processed=policy.processor([{"type":MessageType.EPISODE_STEP.value,"content":content}])
    collated=_rec_to_dtype(policy.collate_fn([processed]),dtype=torch.bfloat16)
    action=collated["inputs"]["action"]; mask=collated["inputs"]["action_mask"]
    horizon=len(joint7); action[:, :horizon, OVERFLOW_SLOT]=torch.as_tensor(joint7,device=action.device,dtype=action.dtype)
    time_mask=mask.any(dim=-1,keepdim=True); mask.zero_(); mask[..., :7]=time_mask; mask[..., OVERFLOW_SLOT]=time_mask.squeeze(-1)
    return collated


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--episode-sequence",type=int,required=True)
    p.add_argument("--checkpoint",type=Path,default=CHECKPOINT); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda")
    p.add_argument("--max-frames",type=int); p.add_argument("--save-meta",action="store_true"); p.add_argument("--no-gradnorm",action="store_true"); args=p.parse_args()
    entry=manifest_entry(args.manifest,args.episode_sequence); policy=Gr00tPolicy(embodiment_tag=EmbodimentTag.ROBOCASA_PANDA_OMRON,model_path=args.checkpoint,device=args.device)
    trajectory=OpenArmTrajectory(entry["dataset_dir"],entry["episode_index"]); frame_count=max(0,len(trajectory)-HORIZON+1)
    if args.max_frames is not None: frame_count=min(frame_count,args.max_frames)
    args.output_dir.mkdir(parents=True,exist_ok=True); losses,cosines,gradnorms=([[] for _ in range(NUM_STEPS)] for _ in range(3)); targets=[]; velocities=[[] for _ in range(NUM_STEPS)]
    for frame in range(frame_count):
        state=right_state(trajectory.state[frame]); actions=right_action(trajectory.actions[frame:frame+HORIZON])
        content=VLAStepData(images={"res256_image_side_0":[trajectory.image("head_image",frame)],"res256_image_side_1":[trajectory.image("left_wrist_image",frame)],"res256_image_wrist_0":[trajectory.image("right_wrist_image",frame)]},masks=None,
            states={"end_effector_position_relative":state[:3][None],"end_effector_rotation_relative":state[3:7][None],"gripper_qpos":np.asarray([[state[7],0]],np.float32),"base_position":np.zeros((1,3),np.float32),"base_rotation":np.zeros((1,4),np.float32)},
            actions={"end_effector_position":actions[:,:3],"end_effector_rotation":actions[:,3:6],"gripper_close":actions[:,7:8],"base_motion":np.zeros((HORIZON,4),np.float32),"control_mode":np.zeros((HORIZON,1),np.float32)},text=trajectory.prompt,embodiment=EmbodimentTag.ROBOCASA_PANDA_OMRON)
        result=policy.model.static_inference(**build_inputs(policy,content,actions[:,6]),compute_gradnorm=not args.no_gradnorm)
        for step in range(NUM_STEPS): losses[step].append(result["final_loss"][step]); cosines[step].append(result["cosine"][step]); gradnorms[step].append(result["gradnorm_vision"][step])
        if args.save_meta:
            targets.append(result["u"][0,:HORIZON,[0,1,2,3,4,5,6,OVERFLOW_SLOT]].numpy())
            for step in range(NUM_STEPS): velocities[step].append(result["v"][step][0,:HORIZON,[0,1,2,3,4,5,6,OVERFLOW_SLOT]].numpy())
    for step in range(NUM_STEPS):
        np.save(args.output_dir/f"final_loss_{step}.npy",np.asarray(losses[step],np.float32)); np.save(args.output_dir/f"cosine_{step}.npy",np.asarray(cosines[step],np.float32)); np.save(args.output_dir/f"gradnorm_vision_step_{step}.npy",np.asarray(gradnorms[step],np.float32))
    if args.save_meta:
        meta=args.output_dir/"meta"; meta.mkdir(exist_ok=True); np.save(meta/"u.npy",np.stack(targets).astype(np.float32))
        for step in range(NUM_STEPS): np.save(meta/f"v_{step}.npy",np.stack(velocities[step]).astype(np.float32))
    (args.output_dir/"trajectory_meta.json").write_text(json.dumps({"model":"gr00t-n1.6","dataset_name":entry["dataset_name"],"episode_index":entry["episode_index"],"episode_sequence":args.episode_sequence,"source_length":len(trajectory),"action_horizon":HORIZON,"num_frames_used":frame_count,"side":"right","metric_dims":[0,1,2,3,4,5,6,12],"overflow_identity_slot":12,"camera_mapping":{"base":"head_image","primary_wrist":"right_wrist_image","secondary":"left_wrist_image"}},indent=2)+"\n")


if __name__=="__main__": main()
