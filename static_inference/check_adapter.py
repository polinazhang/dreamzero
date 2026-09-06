"""Exercise the real dataset and checkpoint transforms without model weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from groot.vla.data.schema import DatasetMetadata
from .robocasa import Episode, dataset_path, RoboCasaDroidAdapter


def main():
    root=Path(__file__).resolve().parents[3]
    checkpoint=root/'models/dreamzero/checkpoints/DreamZero-DROID/experiment_cfg'
    cfg=OmegaConf.load(checkpoint/'conf.yaml')
    metadata=DatasetMetadata.model_validate(json.loads((checkpoint/'metadata.json').read_text())['oxe_droid'])
    transform=instantiate(cfg.transforms.oxe_droid)
    transform.set_metadata(metadata)
    transform.eval()
    head=SimpleNamespace(action_horizon=24,device=torch.device('cpu'),model=SimpleNamespace(action_dim=32))
    policy=SimpleNamespace(trained_model=SimpleNamespace(action_head=head),train_cfg=cfg,
                           eval_transform=transform,modality_configs=instantiate(cfg.modality_configs.oxe_droid),
                           model_dir=checkpoint.parent,embodiment_tag=metadata.embodiment_tag)
    adapter=RoboCasaDroidAdapter(policy)
    episode=Episode(dataset_path(root/'datasets/robocasa/v1.0/target/atomic',2),0)
    sample=adapter.sample(episode,0)
    assert sample.action_mask[0,0].nonzero().flatten().tolist()==[0,1,2,3,4,5,7]
    assert torch.count_nonzero(sample.inputs.state[...,9:])==0
    assert sample.inputs.state_mask[0,0,8]
    assert sample.target_images.shape[1]==25
    assert sample.inputs.images.shape[1]==1
    assert torch.equal(sample.inputs.images,sample.target_images[:,:1])
    assert torch.isfinite(sample.inputs.state).all() and torch.isfinite(sample.action).all()
    print('ADAPTER_PASS',sample.inputs.state.shape,sample.action.shape,sample.inputs.images.shape,sample.target_images.shape,flush=True)

if __name__=='__main__':
    main()
