"""Named checkpoint transforms for explicit dataset/checkpoint slot mappings."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from groot.vla.data.schema import DatasetMetadata

from .core import StaticSample


@dataclass(frozen=True)
class JointMapping:
    dataset: str
    embodiment: str
    state_slots: tuple[int, ...]  # source coordinate i -> checkpoint slot
    action_slots: tuple[int, ...]
    cameras: tuple[tuple[str, str], ...]  # checkpoint field -> source camera
    language_key: str

    @property
    def action_dims(self):
        return tuple(sorted(self.action_slots))


class JointDatasetAdapter:
    """The mapping is explicit; transforms and tensor sizes come from checkpoint."""
    def __init__(self, policy, mapping):
        if policy.embodiment_tag.value != mapping.embodiment:
            raise ValueError(f'{mapping.dataset} requires {mapping.embodiment}; got {policy.embodiment_tag.value}')
        self.policy, self.mapping = policy, mapping
        self.head = policy.trained_model.action_head
        self.transform = copy.deepcopy(policy.eval_transform)
        self.metadata = DatasetMetadata.model_validate(json.loads(
            (Path(policy.model_dir)/'experiment_cfg/metadata.json').read_text())[mapping.embodiment])
        concat = next(t for t in self.transform.transforms if hasattr(t,'state_concat_order'))
        self.state_groups = self._groups(concat.state_concat_order, 'state')
        self.action_groups = self._groups(concat.action_concat_order, 'action')
        self.horizon = self.head.action_horizon
        self.video_offsets = np.asarray(policy.modality_configs.video.delta_indices,dtype=int)
        self.action_offsets = np.asarray(policy.modality_configs.action.delta_indices,dtype=int)[:self.horizon]
        self.required_future = max(int(self.video_offsets.max()),int(self.action_offsets.max()))
        if len(self.action_offsets) != self.horizon:
            raise ValueError('Checkpoint action offsets do not cover its inference horizon')
        self._camera_shapes = None
        self.action_dims = mapping.action_dims
        self._validate_slots(mapping.state_slots, max(b for _,a,b in self.state_groups), 'state')
        self._validate_slots(mapping.action_slots, max(b for _,a,b in self.action_groups), 'action')
        self.action_transforms = [t for t in self.transform.transforms
                                  if hasattr(t,'_normalizers') and any(k.startswith('action.') for k in t._normalizers)]
        if not self.action_transforms:
            raise ValueError('No original checkpoint action normalizers')

    def _groups(self, keys, modality):
        result, start = [], 0
        for key in keys:
            width = int(getattr(self.metadata.modalities,modality)[key.split('.',1)[1]].shape[0])
            result.append((key,start,start+width))
            start += width
        return result

    @staticmethod
    def _validate_slots(slots, width, kind):
        if len(set(slots)) != len(slots) or min(slots)<0 or max(slots)>=width:
            raise ValueError(f'Invalid {kind} mapping {slots} for checkpoint width {width}')

    def map_raw(self,state,actions):
        m = self.mapping
        if state.shape != (len(m.state_slots),) or actions.shape != (self.horizon,len(m.action_slots)):
            raise ValueError(f'{m.dataset} source shape mismatch: state={state.shape}, action={actions.shape}')
        ns = np.zeros(max(b for _,a,b in self.state_groups),dtype=state.dtype)
        na = np.zeros((self.horizon,max(b for _,a,b in self.action_groups)),dtype=actions.dtype)
        ns[list(m.state_slots)] = state
        na[:,list(m.action_slots)] = actions
        states = {key:ns[a:b].copy() for key,a,b in self.state_groups}
        actions = {key:na[:,a:b].copy() for key,a,b in self.action_groups}
        if self.policy.train_cfg.get('relative_action',False):
            keys = self.policy.train_cfg.get('relative_action_keys',None)
            for key,value in actions.items():
                subkey = key.split('.',1)[1]
                relative = (subkey in keys) if keys is not None else ('gripper' not in subkey)
                if relative:
                    reference = 'state.'+subkey
                    if reference not in states:
                        raise ValueError(f'Relative field {key} has no reference state')
                    actions[key] = value-states[reference]
        return states,actions

    def sample(self,episode,t):
        if t < 0 or t+self.required_future >= len(episode):
            raise IndexError('Full target window unavailable')
        shapes = tuple((key,*episode.camera_shapes[camera]) for key,camera in self.mapping.cameras)
        if shapes != self._camera_shapes:
            for key,height,width in shapes:
                self.metadata.modalities.video[key.split('.',1)[1]].resolution = (width,height)
            for transform in self.transform.transforms:
                if type(transform).__name__=='VideoCrop':
                    transform.height=transform.width=None
            self.transform.set_metadata(self.metadata)
            self.transform.eval()
            self._camera_shapes=shapes
        states,actions = self.map_raw(episode.state[t],episode.actions[t+self.action_offsets])
        raw = {key:value[None,None] for key,value in states.items()}
        raw.update({key:episode.images(camera,t+self.video_offsets)[None] for key,camera in self.mapping.cameras})
        raw[self.mapping.language_key] = [[episode.instruction(t)]]
        data = self.transform(raw)
        target_images = data['images'].clone()
        data['images'] = data['images'][:,:1].clone()
        sm = torch.zeros_like(data['state'],dtype=torch.bool)
        sm[...,list(self.mapping.state_slots)] = True
        data['state'] = torch.where(sm,data['state'],torch.zeros_like(data['state']))
        data['state_mask'] = sm
        actions = {key:torch.as_tensor(value) for key,value in actions.items()}
        for transform in self.action_transforms:
            actions=transform(actions)
        na = torch.zeros((1,self.horizon,self.head.model.action_dim),dtype=torch.float32)
        for key,a,b in self.action_groups:
            na[0,:,a:b] = actions[key]
        mask=torch.zeros_like(na,dtype=torch.bool)
        mask[...,list(self.action_dims)] = True
        na=torch.where(mask,na,torch.zeros_like(na))
        device=self.head.device
        data={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        return StaticSample(BatchFeature(data),na.to(device),mask.to(device),
                            torch.ones(1,dtype=torch.bool,device=device),target_images.to(device))
