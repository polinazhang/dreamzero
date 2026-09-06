"""Shared real-checkpoint GPU validation for any static sample."""
import json
import torch
from .core import static_dit_forward

def validate_core(core, sample, output):
    """Real-checkpoint parity against original no-grad DiT, plus step-count coverage."""
    import unittest
    from .test_core import CoreTests
    suite = unittest.TestSuite([CoreTests('test_all_supported_step_counts_integrate_constant_flow')])
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise AssertionError('Original scheduler integration failed')
    head = core.head
    with torch.no_grad():
        clip, y, prompts, caches, xa, xv, ua, uv, sv, sa = core.prepare(sample, core.default_steps)
        vt = torch.ones((1, head.num_frame_per_block), device=xv.device, dtype=torch.int64) * sv.timesteps[0]
        at = torch.ones(xa.shape[:2], device=xa.device, dtype=torch.int64) * sa.timesteps[0]
        cross = head._create_crossattn_caches(1, xv.dtype, xv.device)[0]
        for prompt, cache in zip(prompts, caches):
            original = head.model._forward_inference(xv, vt, prompt,
                        head.num_frame_per_block * (xv.shape[-2]//2) * (xv.shape[-1]//2), cache, cross, 1,
                        y=y, clip_feature=clip, action=xa, timestep_action=at,
                        state=sample.inputs.state.to(torch.bfloat16), embodiment_id=sample.inputs.embodiment_id)
            actual = static_dit_forward(head.model, xv, vt, prompt, y, clip, cache,
                                        action=xa, timestep_action=at, state=sample.inputs.state.to(torch.bfloat16),
                                        current_start_frame=1, checkpoint_blocks=False)
            for a, b in zip(original[:2], actual[:2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    del clip, y, prompts, caches, xa, xv, ua, uv, sv, sa, original, actual, cross
    reports = []
    for count in sorted(set([1, max(2, core.default_steps // 2), core.default_steps])):
        if count > core.default_steps:
            continue
        number = 0
        for result in core.evaluate(sample, count):
            for name in ('action_loss','video_loss','cosine_action','cosine_video',
                         'gradnorm_vision_action','gradnorm_vision_video'):
                if not torch.isfinite(result[name]).all():
                    raise AssertionError(f'Nonfinite {name} at {count} steps')
            number += 1
        assert number == count
        reports.append(dict(steps=count, passed=True))
        print(f'VALIDATED steps={count}', flush=True)
    (output / 'validation.json').write_text(json.dumps({'original_dit_bitwise_parity': True, 'step_counts': reports}, indent=2)+'\n')

