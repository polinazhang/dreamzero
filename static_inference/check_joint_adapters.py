"""Real checkpoint transforms, no GPU/model weights; one combination at a time."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from groot.vla.data.schema import DatasetMetadata


def cpu_policy(checkpoint,embodiment):
    checkpoint=Path(checkpoint)
    cfg=OmegaConf.load(checkpoint/'experiment_cfg/conf.yaml')
    metadata=DatasetMetadata.model_validate(json.loads((checkpoint/'experiment_cfg/metadata.json').read_text())[embodiment])
    from .runtime import local_tokenizer
    from groot.vla.model.n1_5.sim_policy import _update_tokenizer_path_in_config
    _update_tokenizer_path_in_config(cfg, local_tokenizer())
    transform=instantiate(cfg.transforms[embodiment])
    transform.set_metadata(metadata)
    transform.eval()
    config=json.loads((checkpoint/'config.json').read_text())
    hcfg=config['action_head_cfg']['config']
    head=SimpleNamespace(action_horizon=hcfg['action_horizon'],device=torch.device('cpu'),dtype=torch.bfloat16,
                         model=SimpleNamespace(action_dim=hcfg['action_dim']))
    return SimpleNamespace(trained_model=SimpleNamespace(action_head=head),train_cfg=cfg,eval_transform=transform,
                           modality_configs=instantiate(cfg.modality_configs[embodiment]),model_dir=checkpoint,
                           embodiment_tag=metadata.embodiment_tag)


def check(policy,adapter,episode):
    mapping=adapter.mapping
    first=adapter.sample(episode,0)
    assert first.action_mask[0,0].nonzero().flatten().tolist()==list(mapping.action_dims)
    assert first.inputs.state_mask[0,0].nonzero().flatten().tolist()==sorted(mapping.state_slots)
    assert first.action_mask.sum().item()==adapter.horizon*len(mapping.action_slots)
    assert torch.count_nonzero(first.inputs.state[~first.inputs.state_mask])==0
    assert torch.count_nonzero(first.action[~first.action_mask])==0
    assert torch.isfinite(first.inputs.state).all() and torch.isfinite(first.action).all()
    assert torch.equal(first.inputs.images,first.target_images[:,:1])
    # Probe a second frame to exercise cached transforms/normalizer dtypes.
    if len(episode)>adapter.required_future+1:
        second=adapter.sample(episode,1)
        assert second.action.shape==first.action.shape
    print('ADAPTER_PASS',mapping.dataset,first.inputs.state.shape,first.action.shape,
          first.target_images.shape,'valid_dims=',mapping.action_dims,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['droid','yam','openarm'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    if args.dataset=='droid':
        from .droid import DroidAdapter, DroidEpisode, inventory
        policy=cpu_policy(root/'models/dreamzero/checkpoints/DreamZero-DROID','oxe_droid')
        episode=DroidEpisode(inventory(root/'datasets/droid_100/1.0.0')[0])
        try: check(policy,DroidAdapter(policy),episode)
        finally: episode.close()
    else:
        from .joint_runner import configurations, entries, open_episode
        spec=configurations(root)[args.dataset]
        policy=cpu_policy(spec['checkpoint'],spec['embodiment'])
        adapter=spec['adapter'](policy)
        for label,path in spec['datasets']:
            entry=entries(args.dataset,path,spec['limit'])[0]
            episode=open_episode(args.dataset,path,entry)
            try: check(policy,adapter,episode)
            finally: episode.close()


if __name__=='__main__':
    main()
