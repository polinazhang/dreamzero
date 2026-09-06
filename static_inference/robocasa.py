"""RoboCasa LeRobot -> DreamZero-DROID adapter; all metric math lives in core."""
import copy
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch

from .core import StaticSample

TASKS = ('CloseBlenderLid', 'CloseFridge', 'CloseToasterOvenDoor', 'CoffeeSetupMug',
         'NavigateKitchen', 'OpenCabinet', 'OpenDrawer', 'OpenStandMixerHead',
         'PickPlaceCounterToCabinet', 'PickPlaceCounterToStove', 'PickPlaceDrawerToCounter',
         'PickPlaceSinkToCounter', 'PickPlaceToasterToCounter', 'SlideDishwasherRack',
         'TurnOffStove', 'TurnOnElectricKettle', 'TurnOnMicrowave', 'TurnOnSinkFaucet')
ACTION_DIMS = (0, 1, 2, 3, 4, 5, 7)
CAMERAS = {'video.exterior_image_1_left': 'observation.images.robot0_agentview_left',
           'video.exterior_image_2_left': 'observation.images.robot0_agentview_right',
           'video.wrist_image_left': 'observation.images.robot0_eye_in_hand'}


def dataset_path(root, task_id):
    if task_id == 5 or not 1 <= task_id <= len(TASKS):
        raise ValueError('Choose task 1..18 excluding task 5 (NavigateKitchen)')
    candidates = sorted((Path(root) / TASKS[task_id - 1]).glob('*/lerobot/meta/info.json'))
    if len(candidates) != 1:
        raise ValueError(f'Expected one LeRobot dataset for {TASKS[task_id - 1]}, found {candidates}')
    return candidates[0].parent.parent


def episodes(root, limit=50):
    if not 1 <= limit <= 50:
        raise ValueError('RoboCasa episode limit must be 1..50')
    entries = [json.loads(line) for line in (Path(root) / 'meta/episodes.jsonl').read_text().splitlines()]
    return sorted(entries, key=lambda e: e['episode_index'])[:limit]


class Episode:
    def __init__(self, root, index):
        root = Path(root)
        self.index = index
        self.info = json.loads((root / 'meta/info.json').read_text())
        self.modality = json.loads((root / 'meta/modality.json').read_text())
        values = dict(episode_index=index, episode_chunk=index // self.info['chunks_size'])
        table = pq.read_table(root / self.info['data_path'].format(**values))
        self.arrays = {name: np.asarray(table[name].to_pylist()) for name in table.column_names}
        self.tasks = {e['task_index']: e['task'] for e in
                      map(json.loads, (root / 'meta/tasks.jsonl').read_text().splitlines())}
        self.videos = {}
        for camera in CAMERAS.values():
            path = root / self.info['video_path'].format(**values, video_key=camera)
            with av.open(str(path)) as container:
                frames, times = [], []
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format='rgb24'))
                    times.append(float(frame.pts * frame.time_base))
            self.videos[camera] = np.stack(frames)
            timestamps = np.asarray(self.arrays['timestamp']).reshape(-1)
            # Nearest video timestamps, as in the original timestamp-based loader.
            times = np.asarray(times)
            right = np.searchsorted(times, timestamps).clip(0, len(times) - 1)
            left = (right - 1).clip(0)
            selected = np.where(abs(times[left] - timestamps) <= abs(times[right] - timestamps), left, right)
            if np.max(abs(times[selected] - timestamps)) > 1 / self.info['fps'] + 1e-4:
                raise ValueError(f'Video timestamp mismatch: {path}')
            self.videos[camera] = self.videos[camera][selected]

    def field(self, modality, key):
        spec = self.modality[modality][key]
        return self.arrays[spec['original_key']][..., spec['start']:spec['end']]

    def __len__(self):
        return len(self.arrays['timestamp'])

    def instruction(self, t):
        key = self.modality['annotation']['human.task_description']['original_key']
        return self.tasks[int(self.arrays[key][t].reshape(-1)[0])]


class RoboCasaDroidAdapter:
    def __init__(self, policy):
        self.policy = policy
        self.transform = copy.deepcopy(policy.eval_transform)
        from groot.vla.data.schema import DatasetMetadata
        metadata_path = Path(policy.model_dir) / 'experiment_cfg/metadata.json'
        self.metadata = DatasetMetadata.model_validate(json.loads(metadata_path.read_text())[policy.embodiment_tag.value])
        self._camera_shapes = None
        self.head = policy.trained_model.action_head
        self.horizon = self.head.action_horizon
        self.video_offsets = np.asarray(policy.modality_configs.video.delta_indices, dtype=int)
        self.action_offsets = np.asarray(policy.modality_configs.action.delta_indices, dtype=int)[:self.horizon]
        self.required_future = max(int(self.video_offsets.max()), int(self.action_offsets.max()))
        self.state_transform = next(t for t in self.transform.transforms
                                    if hasattr(t, '_normalizers') and 'state.gripper_position' in t._normalizers)
        self.action_transform = next(t for t in self.transform.transforms
                                     if hasattr(t, '_normalizers') and 'action.joint_position' in t._normalizers)

    def sample(self, episode, t):
        if t + self.required_future >= len(episode):
            raise IndexError('Full ground-truth window is unavailable; end padding is not used')
        arm = np.concatenate([episode.field('state', 'end_effector_position_relative')[t],
                              episode.field('state', 'end_effector_rotation_relative')[t]])
        fingers = episode.field('state', 'gripper_qpos')[t]
        # Batched raw modalities; original eval transform performs crop, resize,
        # normalization, view layout, prompt formatting and tokenization.
        raw = {key: episode.videos[camera][t + self.video_offsets][None]
               for key, camera in CAMERAS.items()}
        raw.update({'state.joint_position': arm[None, None].copy(),
                    'state.gripper_position': fingers[:1][None, None].copy(),
                    'annotation.language.language_instruction': [[episode.instruction(t)]]})
        shapes = tuple((key, episode.videos[camera].shape[-2], episode.videos[camera].shape[-3])
                       for key, camera in CAMERAS.items())
        if shapes != self._camera_shapes:
            # Change only source camera geometry, keeping original normalization
            # statistics, crop scale, resize target, and evaluation behavior.
            for key, width, height in shapes:
                self.metadata.modalities.video[key.split('.', 1)[1]].resolution = (width, height)
            for transform in self.transform.transforms:
                if type(transform).__name__ == 'VideoCrop':
                    # This transform caches the original source dimensions on
                    # first metadata binding; invalidate that cache on our copy.
                    transform.height = transform.width = None
            self.transform.set_metadata(self.metadata)
            self.transform.eval()
            self._camera_shapes = shapes
        data = self.transform(raw)
        target_images = data['images'].clone()
        data['images'] = data['images'][:, :1].clone()
        # Extra valid gripper coordinate occupies the next unused model slot.
        # Reuse the existing gripper normalizer; no dataset statistics are fitted.
        second = self.state_transform._normalizers['state.gripper_position'].forward(torch.as_tensor(fingers[1:2]))
        data['state'][..., 8] = second.item()
        data['state'][..., 9:] = 0  # zero AFTER normalization
        data['state_mask'][..., 8] = True
        idx = t + self.action_offsets
        commands = np.concatenate([episode.field('action', 'end_effector_position')[idx],
                                   episode.field('action', 'end_effector_rotation')[idx]], axis=-1)
        native_arm = np.zeros((self.horizon, 7), dtype=commands.dtype)
        native_arm[:, :6] = commands
        if self.policy.train_cfg.get('relative_action', False):
            keys = self.policy.train_cfg.get('relative_action_keys', None)
            if keys is None or 'joint_position' in keys:
                native_arm -= arm
        actions = self.action_transform({'action.joint_position': torch.as_tensor(native_arm),
                                         'action.gripper_position': torch.as_tensor(episode.field('action', 'gripper_close')[idx])})
        action = torch.zeros((1, self.horizon, self.head.model.action_dim), dtype=torch.float32)
        action[0, :, :7] = actions['action.joint_position']
        action[0, :, 7:8] = actions['action.gripper_position']
        mask = torch.zeros_like(action, dtype=torch.bool)
        mask[..., list(ACTION_DIMS)] = True
        action *= mask
        device = self.head.device
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        from transformers.feature_extraction_utils import BatchFeature
        return StaticSample(BatchFeature(data), action.to(device), mask.to(device),
                            torch.ones(1, dtype=torch.bool, device=device), target_images.to(device))
