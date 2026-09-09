"""Common runner for DROID, YAM, and all four OpenArm datasets."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from .core import StaticInferenceCore
from .runtime import load_policy
from .storage import EpisodeWriter


def configurations(root):
    from .franka import DATA_ROOT, FrankaAdapter
    from .droid import DroidAdapter
    from .yam import YamAdapter
    from .openarm import DATASETS, OpenArmAdapter
    checkpoints=root/'models/dreamzero/checkpoints'
    return {
        'franka':dict(checkpoint=checkpoints/'DreamZero-DROID',embodiment='oxe_droid',adapter=FrankaAdapter,
                      limit=3,datasets=[(p.parent.parent.name,p.parent.parent) for p in sorted(DATA_ROOT.glob('*/meta/info.json'))]),
        'droid':dict(checkpoint=checkpoints/'DreamZero-DROID',embodiment='oxe_droid',adapter=DroidAdapter,
                     limit=100,datasets=[('droid',root/'datasets/droid_100/1.0.0')]),
        'yam':dict(checkpoint=checkpoints/'DreamZero-AgiBot',embodiment='agibot',adapter=YamAdapter,
                   limit=100,datasets=[('yam',root/'datasets/molmoact2/MolmoAct2-BimanualYAM-Dataset-500')]),
        'openarm':dict(checkpoint=checkpoints/'DreamZero-AgiBot',embodiment='agibot',adapter=OpenArmAdapter,
                       limit=20,datasets=[(name,root/'datasets/OpenArm'/name) for name in DATASETS]),
    }


def entries(dataset,path,limit):
    if dataset=='droid':
        from .droid_source import inventory
    else:
        from .lerobot_v3 import inventory
    return inventory(path,limit)


def open_episode(dataset,path,entry):
    if dataset=='droid':
        from .droid_source import DroidEpisode
        return DroidEpisode(entry)
    if dataset=='franka':
        from .franka import FrankaEpisode
        return FrankaEpisode(path,entry)
    if dataset=='yam':
        from .yam import YamEpisode
        return YamEpisode(path,entry)
    if dataset=='openarm':
        from .openarm import OpenArmEpisode
        return OpenArmEpisode(path,entry)
    raise ValueError(dataset)


def boolean(value):
    if isinstance(value,bool):
        return value
    if value.lower() in ('true','1','yes'):
        return True
    if value.lower() in ('false','0','no'):
        return False
    raise argparse.ArgumentTypeError('Expected True or False')


def main(dataset):
    root=Path(__file__).resolve().parents[3]
    spec=configurations(root)[dataset]
    parser=argparse.ArgumentParser(description=f'{dataset} static inference using the shared DreamZero core')
    parser.add_argument('--checkpoint',type=Path,default=spec['checkpoint'])
    parser.add_argument('--dataset-root',type=Path,help='For OpenArm, parent of all four named task directories')
    parser.add_argument('--output-root',type=Path,default=root/'results/dreamzero-static'/dataset)
    parser.add_argument('--steps',type=int,default=1,help='Any count from 1 through checkpoint default; actual-run default is 1')
    parser.add_argument('--max-episodes',type=int,default=spec['limit'],help='Debug limit per dataset')
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--save_meta',type=boolean,nargs='?',const=True,default=True)
    parser.add_argument('--preflight',action='store_true',help='Validate all selected source addresses without loading weights')
    args=parser.parse_args()
    if not 1<=args.max_episodes<=spec['limit'] or args.steps<1:
        parser.error(f'--max-episodes must be 1..{spec["limit"]} and --steps must be positive')
    if args.max_frames is not None and args.max_frames<1:
        parser.error('--max-frames must be positive')
    datasets=spec['datasets']
    if args.dataset_root:
        datasets=[(label,args.dataset_root/label if dataset in ('openarm','franka') else args.dataset_root) for label,_ in datasets]
    if not datasets:
        raise FileNotFoundError(f'No dataset subsets found for {dataset}')
    selected=[(label,path,entries(dataset,path,args.max_episodes)) for label,path in datasets]
    inventory=[dict(dataset=label,path=str(path),episodes=rows) for label,path,rows in selected]
    print(json.dumps({'dataset':dataset,'checkpoint':str(args.checkpoint),'steps':args.steps,
                      'selections':[{**row,'episodes':[e['episode_index'] for e in row['episodes']]} for row in inventory]},indent=2),flush=True)
    if args.preflight:
        return
    output=args.output_root/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    policy=load_policy(args.checkpoint,spec['embodiment'])  # one load, including all four OpenArm subsets
    adapter=spec['adapter'](policy)
    core=StaticInferenceCore(policy.trained_model.action_head)
    if args.steps>core.default_steps:
        raise ValueError(f'--steps must be in 1..{core.default_steps}')
    run_metadata=dict(arguments=vars(args),dataset=dataset,inventory=inventory,mapping=asdict(adapter.mapping),
                      action_dims=adapter.action_dims,checkpoint_default_steps=core.default_steps,
                      action_horizon=adapter.horizon,video_offsets=adapter.video_offsets.tolist(),
                      action_offsets=adapter.action_offsets.tolist())
    (output/'run.json').write_text(json.dumps(run_metadata,default=str,indent=2)+'\n')
    print(f'OUTPUT={output}',flush=True)
    for label,path,rows in selected:
        for entry in rows:
            episode=open_episode(dataset,path,entry)
            try:
                frame_count=max(0,len(episode)-adapter.required_future)
                if args.max_frames is not None:
                    frame_count=min(frame_count,args.max_frames)
                if not frame_count:
                    raise ValueError(f'No complete target windows for {label} episode {episode.index}')
                writer=EpisodeWriter(output/label/f'episode_{episode.index:06d}',frame_count,
                                     adapter.action_dims,args.save_meta)
                for frame in range(frame_count):
                    started=time.monotonic()
                    sample=adapter.sample(episode,frame)
                    for result in core.evaluate(sample,args.steps):
                        writer.write(frame,result)
                        print(f'dataset={label} episode={episode.index} frame={frame} step={result["step"]} '
                              f'action_loss={result["action_loss"].item():.7g} video_loss={result["video_loss"].item():.7g}',flush=True)
                    print(f'frame_seconds={time.monotonic()-started:.3f}',flush=True)
                writer.finish(args.steps,dict(dataset=label,episode_index=episode.index,source=entry,
                                               source_length=len(episode),source_frame_indices=list(range(frame_count)),
                                               action_dims=adapter.action_dims,mapping=asdict(adapter.mapping),
                                               steps=args.steps,action_horizon=adapter.horizon,
                                               action_offsets=adapter.action_offsets.tolist(),video_offsets=adapter.video_offsets.tolist()))
            finally:
                episode.close()
    (output/'COMPLETE').touch()
    torch.distributed.destroy_process_group()
