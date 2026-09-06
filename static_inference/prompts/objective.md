# Objective

Launch static inference on one atomic seen task given the task number on 4 datasets. Datasets are in lerobot format.

Droid: `/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/datasets/droid_100/1.0.0` (you should run all 100 demos, diffusion-step = 1)

Robocasa: `/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/datasets/robocasa/v1.0/target/atomic`. (ignore composite)  (you must only run 50 trajectories per task and exclude task 5 navigate kitchen, diffusion-step = full joint denoising steps matching the default value for each checkpoint)

MolmoAct2: /storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/datasets/molmoact2/MolmoAct2-BimanualYAM-Dataset-500 (you should only run the first 100 demos, diffusion-step = 1)

Openarm: 
20 trajectories from each of
/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/datasets/OpenArm/pick_cup
/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/datasets/OpenArm/pour_ice
/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/datasets/OpenArm/use_spoon
/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/datasets/OpenArm/use_steel_spoon
(you should only run 20 trajectories per dataset, sharing the same model load, diffusion-step = 1)

Your objective is to write code to support static inference runs. For context see static-inference-context.md, cosine-similarity.md, vision-grad-norm.md, robocasa.md.

For launching and the main body of code, you should write inside `/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/models/dreamzero/static_inference`. Since the goals require some change in dreamzero architecture, you're allowed to modify ``/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/models/dreamzero` under restrictions below:

## Restrictions

You should create separate methods for static inference that must NOT interfere with any original training or inference functionality in this codebase. Keep zero interference by making new methods self-contained. Do not modify the bodies of any of these existing methods; do not add branches, flags, or arguments to them. Only add new methods; don't delete or alter old function blocks.

However, adding new functions inside an existing class, or calling shared submodules in a new sequence are fine. The above only prohibits modifications that touch the actual code passed for normal training/inference. Neither adding new functions inside an existing class nor calling shared submodules in a new sequence changes that.


# Implementation

DreamZero pads action chunks to max_action_dim=32 and applies action_mask elementwise to the action flow-matching MSE. When vision grad norm is calculated, use the original DreamZero action / video target. Your implementation should match the original dreamzero loss semantics exactly. **Do not invent a new loss.**

You should not change how the dreamzero code originally generate noise for inference or attempt to generate another noise. For all downstream calculations use the same noise generated in the beginning.

Dreamzero model takes a long time to load. Therefore, for you implementations, All trajectories from the same dataset should use the same model load. The 4 openarm datasets should all share the same model load. For one dataset, there should be exactly one model load. Push the job limit to 1 day for other 3 datasets and 2 days for openarm to let them have enough time to finish.

For state/action mapping, always refer to the rule /storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/static-inference/dimensions/rule.md and also view the complete /storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/static-inference/dimensions for data specs (where yam is the molmoact2-yam dataset). You must follow the rule for each invididual implementation


# Storage

Cosine similarity and vision grad norm will be saved in the same run.

The content inside meta/ are decided by the flag `--save_meta=True` passed in to the static inference script

The latents that should be saved as files are (use these as file names followed by .npy as well):
- meta/u_action.npy
- meta/u_video.npy
- meta/v_action_{diffusion_step_idx}.npy
- meta/v_video_{diffusion_step_idx}.npy

- action_loss_{diffusion_step_idx}.npy
- video_loss_{diffusion_step_idx}.npy

- cosine_action_{diffusion_step_idx}.npy
- cosine_video_{diffusion_step_idx}.npy

- gradnorm_vision_action_step_{diffusion_step_idx}.npy
- gradnorm_vision_video_step_{diffusion_step_idx}.npy

Note that here diffusion_step_idx refers to the number of inference steps, set from dreamzero default.
stored u_action/v_action should be masked to real dims.

All values are calculated per frame. When a rollout episode finishes (corresponds to one demo trajectory used), they should be stacked together and saved as npy files. The stacking mechanism should be described in a documentation so other users upon viewing can know exactly what is what.

The content inside /meta should be decided by the flag `--save_meta=True` passed in to the static inference script. This flag should be default True

Results should be written to subfolders with timestamps inside `/storage/home/hcoda1/5/xzhang3205/scratch/vla-adaptation/results/dreamzero-static/{dataset-name}`