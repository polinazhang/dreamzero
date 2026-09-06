"""Shared DreamZero static inference; no dataset, embodiment or checkpoint constants.

Inputs are already in the checkpoint's padded, normalized model space. Every
call evaluates one independent demonstration frame and its future chunk.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class StaticSample:
    inputs: object  # BatchFeature with images, text, state, embodiment_id
    action: torch.Tensor  # [B,H,D], normalized, padded
    action_mask: torch.Tensor
    has_real_action: torch.Tensor
    target_images: torch.Tensor  # [B,T,H,W,C], original transform output


def flow_losses(action_pred, video_pred, action_target, video_target, mask,
                has_real_action, scheduler, action_time, video_time):
    """The reductions/weights in WANPolicyHead.forward, without a new loss."""
    video_target = video_target[..., :video_pred.shape[3], :video_pred.shape[4]]
    if video_pred.shape != video_target.shape:
        raise ValueError(f"video target {video_target.shape} != prediction {video_pred.shape}")
    video_mse = F.mse_loss(video_pred.float(), video_target.float(), reduction="none").mean((1, 3, 4))
    vw = scheduler.training_weight(video_time.flatten()).reshape(video_time.shape).to(video_pred.device)
    video_loss = (video_mse * vw).mean()
    action_mse = F.mse_loss(action_pred.float(), action_target.float(), reduction="none") * mask
    action_mse = has_real_action.reshape(-1, 1, 1).float() * action_mse
    aw = scheduler.training_weight(action_time.flatten()).reshape(action_time.shape).to(action_pred.device)
    action_loss = (action_mse.mean(dim=2) * aw).mean()
    return action_loss, video_loss


def cosine(pred, target, mask=None, eps=1e-6):
    pred, target = pred.float(), target.float()
    if mask is not None:
        pred, target = pred * mask, target * mask
    pred, target = pred.flatten(1), target.flatten(1)
    return (pred * target).sum(1) / (pred.norm(dim=1) * target.norm(dim=1) + eps)


def joint_norm(grads):
    return torch.sqrt(sum(g.float().square().sum() for g in grads if g is not None))


def static_dit_forward(model, x, timestep, context, y, clip_feature, kv_cache,
                       action=None, timestep_action=None, state=None,
                       current_start_frame=0, checkpoint_blocks=True):
    """Additive grad-capable version of _forward_inference/_forward_blocks.

    Uses the exact original submodules and attention path. Block checkpointing
    only recomputes activations; the warmed KV tensors are immutable inputs.
    """
    from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import sinusoidal_embedding_1d
    if y is not None and model.concat_first_frame_latent:
        x = torch.cat([x, y.to(x.dtype)], dim=1)
    x = model.patch_embedding(x)
    grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
    freqs = model._create_freqs(grid_size=grid_size, start_frame=current_start_frame)
    x = x.flatten(2).transpose(1, 2)
    batch, seq_len = x.shape[:2]
    frames = timestep.shape[1]
    action_length, register_length = 0, None
    embodiment = torch.zeros(batch, dtype=torch.long, device=x.device)
    if action is not None:
        af = model.action_encoder(action, timestep_action, embodiment)
        sf = model.state_encoder(state, embodiment)
        registers = torch.cat([af, sf], dim=1)
        action_length, register_length = af.shape[1], registers.shape[1]
        x = torch.cat([x, registers], dim=1)
    if frames <= seq_len:
        timestep = timestep.repeat_interleave((seq_len + frames - 1) // frames, dim=1)[:, :seq_len]
    else:
        indices = torch.linspace(0, frames - 1, seq_len, device=timestep.device, dtype=torch.long)
        timestep = timestep[:, indices]
    if action is not None:
        timestep = torch.cat([timestep, timestep_action, timestep_action[:, ::timestep_action.shape[1] // sf.shape[1]]], 1)
    e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, timestep.flatten()).type_as(x))
    e = e.unflatten(0, (batch, -1))
    e0 = model.time_projection(e).unflatten(2, (6, model.dim))
    context = model.text_embedding(context)
    if clip_feature is not None:
        context = torch.cat([model.img_emb(clip_feature), context], 1)
    updated = []
    for index, block in enumerate(model.blocks):
        kwargs = dict(e=e0, freqs=freqs, freqs_action=model.freqs_action,
                      freqs_state=model.freqs_state, context=context,
                      action_register_length=register_length, kv_cache=kv_cache[index],
                      current_start_frame=current_start_frame)
        if torch.is_grad_enabled() and checkpoint_blocks:
            # Discard returned cache during differentiated calls: it is never
            # fed back into this frame's fixed conditioning.
            def run_block(value, module=block, block_kwargs=kwargs):
                return module(x=value, **block_kwargs)[0]
            x = checkpoint(run_block, x, use_reentrant=False)
        else:
            x, cache = block(x=x, **kwargs)
            updated.append(cache.detach())
    ap = model.action_decoder(x[:, seq_len:seq_len + action_length], embodiment) if action is not None else None
    vp = model.unpatchify(model.head(x[:, :seq_len], e[:, :seq_len].unsqueeze(2)), grid_size)
    return vp, ap, updated


@contextmanager
def attention_backward_context(model):
    """Have cuDNN retain backward auxiliaries without enabling model dropout.

    DreamZero's TE wrapper passes its module.training flag to fused_attn.
    Eval mode omits backward auxiliaries even when autograd is enabled. Only
    zero-dropout fused attention wrappers are temporarily enabled here, and
    every flag is restored before control returns to the caller.
    """
    changed = []
    try:
        for module in model.modules():
            if (type(module).__name__ == "FusedAttention" and
                    type(module).__module__.endswith("cudnn_attention")):
                if module.attention_dropout != 0:
                    raise ValueError("Static cuDNN gradients require zero attention dropout")
                changed.append((module, module.training))
                module.training = True
        yield
    finally:
        for module, training in changed:
            module.training = training


class StaticInferenceCore:
    def __init__(self, head, checkpoint_blocks=True):
        self.head = head
        self.checkpoint_blocks = checkpoint_blocks
        self.default_steps = int(head.num_inference_steps)
        self._prompt_cache = {}

    def schedulers(self, steps, device):
        from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler
        if not isinstance(steps, int) or not 1 <= steps <= self.default_steps:
            raise ValueError(f"steps must be in 1..{self.default_steps}, got {steps}")
        result = []
        for _ in range(2):
            scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.head.scheduler.num_train_timesteps,
                                                   shift=1, use_dynamic_shifting=False)
            scheduler.set_timesteps(steps, device=device, shift=self.head.sigma_shift)
            result.append(scheduler)
        video, action = result
        if self.head.config.decouple_inference_noise:
            final = self.head.config.video_inference_final_noise
            maximum = video.sigmas[0].item()
            video.sigmas = video.sigmas * (maximum - final) / maximum + final
            video.timesteps = (video.sigmas[:-1] * 1000).to(torch.int64)
        return video, action

    def prepare_video(self, images, target=False):
        head = self.head
        video = images.permute(0, 4, 1, 2, 3)
        if video.dtype == torch.uint8:
            video = video.float() / 255
            if not target:
                video = video.to(head.dtype)
            b, c, t, h, w = video.shape
            video = head.normalize_video(video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
            video = video.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        video = video.to(torch.bfloat16)
        height = getattr(head.config, "target_video_height", None)
        width = getattr(head.config, "target_video_width", None)
        if height is None or width is None:
            if getattr(head.model, "frame_seqlen", None) in (50, 55):
                height, width = 176, 320
        if height is not None and width is not None and video.shape[-2:] != (height, width):
            b, c, t, h, w = video.shape
            # Match the original head's reshape and interpolation exactly.
            video = F.interpolate(video.reshape(b * t, c, h, w), size=(height, width),
                                  mode="bilinear", align_corners=False).reshape(b, c, t, height, width)
        return video

    @torch.no_grad()
    def prepare(self, sample, steps):
        head = self.head
        data = sample.inputs
        if data.state.shape[0] != 1:
            raise ValueError("Evaluate one frame at a time so per-example gradients retain original loss scaling")
        video = self.prepare_video(data.images)
        if video.shape[2] != 1:
            raise ValueError("Each static sample must provide its current observed frame (T=1)")
        b, _, _, h, w = video.shape
        key = tuple((name, data[name].cpu().numpy().tobytes()) for name in
                    ("text", "text_attention_mask", "text_negative", "text_attention_mask_negative"))
        if key not in self._prompt_cache:
            # Bounded cache: reuse a task's prompt without accumulating GPU tensors.
            self._prompt_cache.clear()
            self._prompt_cache[key] = [head.encode_prompt(t, m).detach() for t, m in head._prepare_text_inputs(data)]
        prompts = self._prompt_cache[key]
        clip, y, image = head.encode_image(video[:, :, :1].transpose(1, 2), head.num_frames, h, w)
        clip, y = clip.to(image.dtype).detach(), y.detach()
        noise_v = head.generate_noise((b, image.shape[1], head.num_frame_per_block, image.shape[3], image.shape[4]),
                                     seed=head.seed, device="cuda", dtype=torch.bfloat16)
        noise_a = head.generate_noise((b, head.action_horizon, head.model.action_dim),
                                     seed=head.seed, device="cuda", dtype=torch.bfloat16)
        target_video = self.prepare_video(sample.target_images, target=True)
        target_z = head.encode_video(target_video, head.tiled,
                                     (head.tile_size_height, head.tile_size_width),
                                     (head.tile_stride_height, head.tile_stride_width))
        target_z = target_z[:, :, 1:1 + head.num_frame_per_block]
        if target_z.shape != noise_v.shape:
            raise ValueError(f"Encoded future target {target_z.shape} does not match inference noise {noise_v.shape}")
        if sample.action.shape != noise_a.shape:
            raise ValueError(f"Action target {sample.action.shape} != noise {noise_a.shape}")
        target_a = head.scheduler.training_target(sample.action.to(dtype=head.dtype), noise_a, None)
        target_v = head.scheduler.training_target(target_z, noise_v, None)
        empty = head._create_kv_caches(b, noise_v.dtype, noise_v.device,
                                      (image.shape[-2] // 2) * (image.shape[-1] // 2))
        caches = []
        for index, prompt in enumerate(prompts):
            _, _, cache = static_dit_forward(head.model, image, torch.zeros((b, 1), device=image.device, dtype=torch.int64),
                                             prompt, y[:, :, :1], clip, empty[index], checkpoint_blocks=False)
            caches.append(cache)
        sv, sa = self.schedulers(steps, noise_v.device)
        return clip, y[:, :, 1:1 + head.num_frame_per_block], prompts, caches, noise_a, noise_v, target_a, target_v, sv, sa

    def evaluate(self, sample: StaticSample, steps=None) -> Iterator[dict]:
        steps = self.default_steps if steps is None else steps
        # Validate before allocating tensors or running encoders.
        if not isinstance(steps, int) or not 1 <= steps <= self.default_steps:
            raise ValueError(f"steps must be in 1..{self.default_steps}")
        head = self.head
        if head.ip_size != 1:
            raise ValueError("Static gradients require a single-process model; inference P2P is not differentiable")
        clip, y, prompts, caches, xa, xv, ua, uv, sv, sa = self.prepare(sample, steps)
        for k, tv in enumerate(sv.timesteps):
            ta = sa.timesteps[k]
            at = torch.ones(xa.shape[:2], dtype=torch.int64, device=xa.device) * ta
            vt = torch.ones((xv.shape[0], xv.shape[2]), dtype=torch.int64, device=xv.device) * tv
            with attention_backward_context(head.model), torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                dc = torch.zeros_like(clip, requires_grad=True)
                dy = torch.zeros_like(y, requires_grad=True)
                predictions = [static_dit_forward(head.model, xv.detach(), vt, prompt, y + dy, clip + dc,
                                                 cache, action=xa.detach(), timestep_action=at,
                                                 state=sample.inputs.state.to(torch.bfloat16), current_start_frame=1,
                                                 checkpoint_blocks=self.checkpoint_blocks)[:2]
                               for prompt, cache in zip(prompts, caches)]
                vp, ap = predictions[0]
                if len(predictions) == 2:
                    vp = predictions[1][0] + head.cfg_scale * (vp - predictions[1][0])
                la, lv = flow_losses(ap, vp, ua, uv, sample.action_mask, sample.has_real_action,
                                     head.scheduler, at, vt)
                ga = torch.autograd.grad(la, (dc, dy), retain_graph=True, allow_unused=True)
                gv = torch.autograd.grad(lv, (dc, dy), allow_unused=True)
                out = dict(step=k, action_loss=la.detach(), video_loss=lv.detach(),
                           cosine_action=cosine(ap.detach(), ua, sample.action_mask).detach(),
                           cosine_video=cosine(vp.detach(), uv[..., :vp.shape[3], :vp.shape[4]]).detach(),
                           gradnorm_vision_action=joint_norm(ga).detach(), gradnorm_vision_video=joint_norm(gv).detach(),
                           u_action=ua.detach(), u_video=uv.detach(), v_action=ap.detach(), v_video=vp.detach(),
                           action_timestep=ta.detach(), video_timestep=tv.detach())
                if not sample.has_real_action.item():
                    out["cosine_action"] = torch.full_like(out["cosine_action"], float("nan"))
            with torch.no_grad():
                xv = sv.step(out["v_video"].transpose(1, 2), tv, xv.transpose(1, 2), step_index=k, return_dict=False)[0].transpose(1, 2).detach()
                xa = sa.step(out["v_action"], ta, xa, step_index=k, return_dict=False)[0].detach()
            # Release each step's graph before yielding to the streaming writer.
            del predictions, vp, ap, la, lv, dc, dy, ga, gv
            yield out
