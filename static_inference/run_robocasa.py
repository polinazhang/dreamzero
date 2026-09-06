"""RoboCasa runner using the checkpoint/dataset-independent static core."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch

from .core import StaticInferenceCore, static_dit_forward
from .robocasa import ACTION_DIMS, TASKS, Episode, RoboCasaDroidAdapter, dataset_path, episodes
from .runtime import load_policy
from .storage import EpisodeWriter


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected True or False')


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=root / 'models/dreamzero/checkpoints/DreamZero-DROID')
    parser.add_argument('--dataset-root', type=Path, default=root / 'datasets/robocasa/v1.0/target/atomic')
    parser.add_argument('--output-root', type=Path, default=root / 'results/dreamzero-static/robocasa')
    parser.add_argument('--task-id', type=int, nargs='+', default=[2])
    parser.add_argument('--steps', type=int, default=None, help='1 through checkpoint default; omitted = full default')
    parser.add_argument('--max-episodes', type=int, default=50)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--save_meta', type=boolean, nargs='?', const=True, default=True)
    parser.add_argument('--preflight', action='store_true', help='Inspect datasets without loading model')
    parser.add_argument('--validate-core', action='store_true', help='Compare original forward and test 1/intermediate/default on first frame')
    args = parser.parse_args()
    datasets = [(i, dataset_path(args.dataset_root, i)) for i in args.task_id]
    selected = [(i, path, episodes(path, args.max_episodes)) for i, path in datasets]
    inventory = [dict(task_id=i, task=TASKS[i-1], dataset=str(path),
                      episodes=[e['episode_index'] for e in entries],
                      frames=sum(e['length'] for e in entries)) for i, path, entries in selected]
    print(json.dumps(inventory, indent=2), flush=True)
    if args.preflight:
        return
    output = args.output_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    (output / 'run.json').write_text(json.dumps({'arguments': vars(args), 'inventory': inventory}, default=str, indent=2) + '\n')
    print(f'OUTPUT={output}', flush=True)
    policy = load_policy(args.checkpoint, 'oxe_droid')  # exactly one model load
    core = StaticInferenceCore(policy.trained_model.action_head)
    adapter = RoboCasaDroidAdapter(policy)
    steps = core.default_steps if args.steps is None else args.steps
    if not 1 <= steps <= core.default_steps:
        raise ValueError(f'--steps must be in 1..{core.default_steps}')
    validation_done = False
    for task_id, path, entries in selected:
        for entry in entries:
            episode = Episode(path, entry['episode_index'])
            frame_count = max(0, len(episode) - adapter.required_future)
            if args.max_frames is not None:
                frame_count = min(frame_count, args.max_frames)
            if frame_count == 0:
                raise ValueError(f'No full target windows for task {task_id}, episode {episode.index}')
            writer = EpisodeWriter(output / f'task_{task_id:02d}' / f'episode_{episode.index:06d}',
                                   frame_count, ACTION_DIMS, args.save_meta)
            for t in range(frame_count):
                started = time.monotonic()
                sample = adapter.sample(episode, t)
                if args.validate_core and not validation_done:
                    validate_core(core, sample, output)
                    validation_done = True
                for result in core.evaluate(sample, steps):
                    writer.write(t, result)
                    print(f'task={task_id} episode={episode.index} frame={t} step={result["step"]} '
                          f'action_loss={result["action_loss"].item():.7g} video_loss={result["video_loss"].item():.7g}', flush=True)
                print(f'frame_seconds={time.monotonic()-started:.3f}', flush=True)
            writer.finish(steps, dict(task_id=task_id, episode_index=episode.index, frame_count=frame_count,
                                      source_length=len(episode), source_frame_indices=list(range(frame_count)),
                                      action_dims=ACTION_DIMS, steps=steps, action_horizon=adapter.horizon,
                                      video_offsets=adapter.video_offsets.tolist(), checkpoint=str(args.checkpoint)))
    (output / 'COMPLETE').touch()
    torch.distributed.destroy_process_group()


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


if __name__ == '__main__':
    main()
