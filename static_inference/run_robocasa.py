"""RoboCasa runner using the checkpoint/dataset-independent static core."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch

from .core import StaticInferenceCore
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


if __name__ == '__main__':
    main()
