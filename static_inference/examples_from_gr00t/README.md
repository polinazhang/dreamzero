# Static Inference for GR00T-N1.6

Analysis-only mode (see `prompts/static-inference-context.md`): given RoboCasa
demonstration frames, the model runs its standard 4-step flow-matching denoising
loop starting from a single noise tensor (drawn exactly like
`get_action_with_features`) and never executes actions. The per-step velocity
predictions are compared against the ground-truth action chunk.

- Model: `/coc/testnvme/xzhang3205/vla-adaptation/checkpoints/gr00t/gr00t-n1.6`
  with its existing `statistics.json` (never recomputed). Embodiment
  `ROBOCASA_PANDA_OMRON` (embodiment_id 13).
- Datasets:
  `/coc/testnvme/xzhang3205/vla-adaptation/datasets/robocasa365/atomic-seen-splits/task_{i}_demo_50`,
  `i` in 1..18.
- Padding: the checkpoint pads actions to `max_action_horizon=50`,
  `max_action_dim=128`; the real action chunk is horizon 16 x dim 12 (robocasa
  action keys), covered by `action_mask`. All losses/cosines/grad norms are
  masked to the real dims.

## Running

```bash
# single task locally
python static_inference/run_static_inference.py --task_id 2 --save_meta

# debug: 1 episode, 2 frames
python static_inference/run_static_inference.py --task_id 2 \
    --max_episodes 1 --max_frames 2 --save_meta

# slurm: one job per task (1 node, 32 cpus, 4x A40, qos long, 3 days, kira-lab)
python static_inference/launch_static_inference.py --task_id 2 --extra_args "--save_meta"
python static_inference/launch_static_inference.py --all --extra_args "--save_meta"
```

Slurm logs: `/coc/testnvme/xzhang3205/vla-adaptation/slurms/`. Generated sbatch
scripts: `static_inference/generated_sbatch/`.

## Output layout

```
<coc/testnvme/xzhang3205/static/gr00t>/<timestamp>/task_<i>/
    summary.json                       per-task means over all frames
    episode_<j:06d>/                   j = episode index in the LeRobot loader
        final_loss_{n}.npy             (T_ep,) float32, ALWAYS saved
        cosine_{n}.npy                 (T_ep,) float32, ALWAYS saved
        gradnorm_vision_step_{n}.npy   (T_ep,) float32, ALWAYS saved
        meta/                          only with --save_meta
            u.npy                      (T_ep, 16, 12) float32
            v_{n}.npy                  (T_ep, 16, 12) float32
```

`n` in 0..3 is the denoising step index; `<timestamp>` is `%Y%m%d_%H%M%S` of the
run start (one run = one task). `--save_meta` gates ONLY `meta/u.npy` and
`meta/v_{n}.npy`; the scalar files are always written.

## Stacking semantics

- Frames per episode: `t = 0 .. ep_len - 16` inclusive (the full 16-step
  ground-truth action chunk `[t, t+16)` is available — matching training's
  valid steps, no end padding). `T_ep = ep_len - 15`.
- **Axis 0 of every file in an episode directory is the frame index `t`**, and is
  aligned across all files of that episode (`final_loss_2.npy[k]`,
  `cosine_2.npy[k]`, `gradnorm_vision_step_2.npy[k]`, `meta/u.npy[k]`,
  `meta/v_2.npy[k]` all describe frame `t = k`).
- Index `n` is the n-th denoising step at `t_flow = n/4` in gr00t's convention
  (`num_inference_timesteps = 4`, `t_flow = 0, 0.25, 0.5, 0.75`). The latent at
  step 0 is pure noise.
- Per frame, ONE noise tensor `eps ~ N(0, I)` of shape `(50, 128)` (padded
  action space) is drawn exactly as in `get_action_with_features` and reused as
  the `t=0` latent for all 4 steps and all downstream computations.
- `u = gt_action - noise` (gr00t's velocity convention `v = actions - noise`),
  computed in the padded `(50, 128)` normalized action space and cropped to the
  real dims `(16, 12)` for storage.
- `v_{n}` is the model's velocity prediction at denoising step `n` for the same
  noise/latent trajectory, likewise cropped to `(16, 12)`.
- `final_loss_{n}` is the masked MSE between `v_{n}` and `u` over valid elements
  only — `sum((v_n - u)^2 * mask) / (mask.sum() + 1e-6)` — identical semantics
  to the gr00t training loss.
- `cosine_{n}` is the cosine similarity between `v_{n}` and `u` (see
  `prompts/cosine-similarity.md`: the doc's `u_t = eps - A*` is the negative of
  gr00t's `v = actions - noise`, so `cos(v_n, u)` here equals the doc's
  `cos(-v_n, u_t)` and is +1 for a perfect prediction), computed over all valid
  (masked) elements pooled into a single inner product:
  `<v_n*mask, u*mask> / (||v_n*mask|| * ||u*mask|| + 1e-6)`.
- `gradnorm_vision_step_{n}` is `||grad_{h_v} L^{(n)}||_2`, the local
  sensitivity of the step-`n` masked loss to the vision embedding `h_v` (output
  of the image encoder + projector, in LLM embedding space, before the language
  model; all camera views concatenated). Implemented via the additive
  reparameterization `h_v + delta`, `delta = zeros_like(h_v).requires_grad_(True)`,
  `torch.autograd.grad(loss, delta)` — equal to `dL/dh_v` at `delta = 0` (see
  `prompts/vision-grad-norm.md`). All other inputs, including the step-`n`
  latent from the no-grad pass, are held fixed.

## Code structure (additive-only changes)

- `gr00t/model/modules/eagle_backbone.py`
  - `EagleBackbone.extract_vision_embeddings(vl_input)` -> `h_v`
    (`[num_img_tokens, C]`).
  - `EagleBackbone.forward_with_vision_embeds(vl_input, vision_embeds)` -> same
    `BatchFeature` as `forward()`, replicating `Eagle3_VLModel.forward`'s
    embedding-scatter + language-model call with the provided embeddings.
- `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`
  - `Gr00tN1d6ActionHead.static_denoise_step(...)` -> one denoising loop
    iteration (velocity prediction), grad-capable.
  - `Gr00tN1d6.static_inference(inputs, num_inference_steps=None,
    compute_gradnorm=True)` -> per-step dict `{u, v, final_loss, cosine,
    gradnorm_vision, action_mask}` (CPU tensors / Python floats).

None of the existing methods are modified or call the new ones; training and
regular inference paths are untouched.
