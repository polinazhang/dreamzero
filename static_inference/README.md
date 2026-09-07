# DreamZero static inference

`core.py` contains the shared numerical implementation. It has no dataset names,
checkpoint paths, embodiment layouts or run-specific step counts. Adapters supply
normalized padded state/action tensors, an action mask, current observation,
language, and the ground-truth video clip. The loaded head supplies widths,
horizons, visual encoders, transformer weights, noise generation, training loss
weights and inference scheduler settings.

`StaticInferenceCore.evaluate(sample, steps=N)` accepts every integer from 1
through `head.num_inference_steps`; omitting N uses that default. The production
RoboCasa configuration uses the full default. Other dataset adapters can use the
same API with N=1 or any supported intermediate value. DROID, YAM and OpenArm runners default to one step and support the same full range.

All files are new. No original training/inference method is edited, patched or
replaced. Static transformer forward calls the original submodules and uses
non-reentrant activation checkpointing to bound gradient memory. Original
inference's cross-process prediction exchange is not differentiable, so this
runner keeps both guidance branches in one process on one 80 GB-or-larger GPU.
For differentiation, only the original zero-dropout cuDNN fused-attention wrappers
temporarily enable their backward auxiliaries; flags are restored afterward.
All other modules remain in evaluation mode.

## RoboCasa run

From the DreamZero repository, submit:

```bash
sbatch --output=static_inference/run-%j.out --error=static_inference/run-%j.err \
  static_inference/run_robocasa.sbatch --task-id 2
```

The default is task 2 (CloseFridge). Supply multiple task IDs to process them
in the same process/model load; task 5 is rejected. Each task uses its first 50
episodes in episode-index order. The allocation has a one-day limit. `--steps N` is available on this same
runner, not a separate implementation. `--save_meta=False` disables only latent
files; default is True. Debug limits `--max-episodes` and `--max-frames` are for
validation; production defaults do not subsample frames.

A smoke run uses `--max-episodes 1 --max-frames 1`; it executes only the
requested denoising trajectory. No validation sweep or extra model forward runs
inside inference. CPU tests are separate: `.venv/bin/python -m unittest
static_inference.test_core`. A source inventory is available with
`python -m static_inference.run_robocasa --preflight`.

## Mapping and preprocessing

The adapter reads the dataset's `meta/modality.json`, rather than assuming
RoboCasa's environment action order matches LeRobot storage order. DROID's seven
arm-state slots receive the three EEF-position and four quaternion coordinates
unchanged. Gripper-state coordinates occupy model slots 7 and 8; the second uses
the original checkpoint gripper normalizer. Remaining unavailable state slots
are zero after normalization. No inverse kinematics or quaternion conversion is
performed.

The six EEF action-command coordinates map to model slots 0..5 and gripper to 7.
Slot 6 and all other action slots are excluded by the mask. Mapped action fields
use the checkpoint's original relative-action conversion (subtract current
mapped arm state for its relative joint-position field) and original q99
normalizers. Statistics are never fitted or recomputed. Camera assignment is
left agent view -> first exterior, right agent view -> second exterior, wrist ->
wrist. A separate copy of the checkpoint transforms uses the actual source
camera resolution in its metadata, retaining original statistics, crop scale and
resize target. The checkpoint's evaluation transforms perform resizing, cropping,
view composition, language formatting and tokenization.

## Frames, targets and differentiation

Each row evaluates one current demonstration frame independently. The inference
cache is warmed with that frame by the original causal attention computation;
no previous evaluation's generated history is carried to the next row. This is
the original single-observation path, which resets when T=1. The warmed KV
cache stays fixed for every local derivative.

Action offsets and video offsets come from the checkpoint's selected modality
configuration, expressed in dataset frame indices. DROID supplies action offsets
0..23 and video offsets 0..24. Only frames with the entire configured future
window available are evaluated; there is no end padding or temporal resampling.
Thus an episode with L frames contributes L-24 rows. Axis 0 is source frame
`t=0,1,...,L-25`, and is identical across every file.

The entire ground-truth sample clip is transformed and encoded by the original
VAE, with fresh per-call temporal caches. Its first latent is the observation;
the following `num_frame_per_block` latents are the target for the first
inference block. This is not whole-episode latent precomputation/slicing. The
saved video arrays describe this predicted block, not every latent in the
longer training sample clip.

Action and video noise are each generated once per evaluation by the original
`head.generate_noise` calls, with its original seed (1140 in the current head),
dtype, device and shapes. The same tensors initialize denoising and define
`u_action = noise_action - normalized_action` and `u_video = noise_video - Z*`.
Targets retain the original training-path image normalization precision and
model-facing action dtype. There is no additional training-noise sample. Each requested step count configures
the original UniPC schedulers, including the head's shift and decoupling settings.
The original head's `should_run_model` controls prediction computation/reuse.
Launchers do not override its DiT mask or dynamic-cache configuration. A scheduler
step that reuses a flow still receives metrics at its own scheduler timestep;
gradients differentiate that reused flow's original conditioning graph. There
is no additional forward at the skipped step's latent input.

Video flow is `unconditional + cfg_scale * (conditional - unconditional)`;
action flow is the conditional branch, matching the flows sent to the original
schedulers. Losses retain original timestep training weights and reductions:
action squared error is masked, then averaged over the entire padded action
width and horizon; video error is averaged over channels/spatial dimensions and
weighted over latent time. Neither loss divides by the count of valid action
coordinates. At timesteps whose original training weight is zero, the loss and
its gradient norm are correctly zero.

Cosines pool the complete field per evaluation frame, with action masking and
`epsilon=1e-6`. Video target spatial dimensions are cropped to the prediction as
in the original loss. Gradients are taken jointly with respect to additive CLIP
and VAE-conditioning perturbations, including the conditioning tensor returned
by the original image encoder. The norm is the square root of the summed squared
gradients across both tensors. Denoising inputs, ground truth and warmed caches
are held fixed; gradients do not propagate through prior scheduler steps.

## Output arrays

A run creates a timestamped directory under `results/dreamzero-static/robocasa`.
Each task/episode directory contains the following float32 arrays. Here F is the
number of evaluation frames, H the checkpoint action horizon, and k is a
zero-based inference-step index:

| File | Shape |
| --- | --- |
| `action_loss_k.npy`, `video_loss_k.npy` | `[F]` |
| `cosine_action_k.npy`, `cosine_video_k.npy` | `[F]` |
| `gradnorm_vision_action_step_k.npy`, `gradnorm_vision_video_step_k.npy` | `[F]` |
| `meta/u_action.npy`, `meta/v_action_k.npy` | `[F,H,7]` |
| `meta/u_video.npy`, `meta/v_video_k.npy` | `[F,C,T_latent,H_latent,W_latent]` |

Saved action coordinates are compacted to `[0,1,2,3,4,5,7]` in that order. Losses
still use the full padded tensors. Target video spatial shape is retained in
`u_video`; comparison crops it only when required by the original model output.
`episode.json` records the source frame indices, offsets, horizon and step count.

Files are streamed as `.npy.partial` memory maps to avoid accumulating a whole
episode's video fields in RAM. They are renamed to `.npy` only once every
frame/step has been written, followed by an episode `COMPLETE` marker. Failed or
time-limited episodes remain explicitly partial. `run.json` records selection
and arguments, and a run-level `COMPLETE` appears only after every selected episode finishes.

## DROID, YAM and OpenArm

All runners use `core.py`, the same losses, gradients, schedulers and output writer.
Each combination has an explicit mapping following `dimensions/rule.md`:

| Dataset / checkpoint | Source state and action mapping | Selected episodes |
| --- | --- | --- |
| DROID / DreamZero-DROID | Recorded joint positions 0..6 → arm 0..6; gripper → 7 | All 100 in requested TFDS 1.0.0 |
| YAM / DreamZero-AgiBot | Left six joints → 0..5; right six → 7..12; left/right grippers → 14/15 | First 100 by episode index |
| OpenArm / DreamZero-AgiBot | Left seven → 0..6; right seven → 7..13; grippers → 14/15 | First 20 in each of four subsets |

YAM source order is left arm, left gripper, right arm, right gripper. Saved action
coordinates follow ascending model slot order. Thus compact action widths are
8, 14 and 16 respectively, replacing the RoboCasa-specific width 7 above.
Unobserved states are zero **after** checkpoint normalization, and unmatched action
slots are masked out. Original relative-action transforms and normalization
statistics apply to mapped coordinates. No statistics are fitted to these datasets.
DROID uses recorded joint-position actions because that representation matches
its checkpoint. YAM discards each unobserved seventh model arm coordinate.
OpenArm excludes AgiBot head/waist actions and zeroes their unobserved state.

DROID cameras map exterior 1/exterior 2/wrist to their corresponding model views.
YAM top/left/right and OpenArm head/left wrist/right wrist map to AgiBot
head/left hand/right hand. Original evaluation transforms use source image geometry.
DROID reads TFRecord Examples directly without TensorFlow; YAM/OpenArm read
LeRobot v3 episode addresses and video segment timestamps. DROID has L-24 evaluated
frames per episode; AgiBot uses its native 48-action/49-image window and L-48 frames.

From the DreamZero repository:

```bash
sbatch static_inference/run_droid.sbatch
sbatch static_inference/run_yam.sbatch
sbatch static_inference/run_openarm.sbatch
```

DROID and YAM allocations have one-day limits; OpenArm has two days. Each run
loads its model once. The OpenArm run includes pick_cup, pour_ice, use_spoon and
use_steel_spoon in that same process. All accept `--steps N`, `--max-episodes`,
`--max-frames`, and `--save_meta=False`. Output defaults to
`results/dreamzero-static/{dataset}` with the array layout described above.

CPU source preflight: `python -m static_inference.run_droid --preflight` (replace
`droid` with `yam` or `openarm`). Original-transform checks:
`python -m static_inference.check_joint_adapters droid`. Mapping tests:
`python -m unittest static_inference.test_joint_adapters`. These require the
DreamZero environment and its local Hugging Face tokenizer cache. Both runtime
and CPU validation use the original policy's tokenizer-path override to resolve
UMT5 from the local cache, avoiding stale training-machine paths.

The local YAM copy currently lacks `videos/observation.images.left`; preflight
rejects this incomplete source. Restore that camera directory before running YAM.

Runtime initialization follows `socket_test_optimized_AR.main`: TE attention and
`torch._dynamo.config.recompile_limit = 800`, set before policy construction.
The scheduler remains compiled exactly as in the original implementation.
