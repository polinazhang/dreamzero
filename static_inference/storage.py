"""Stream frame-aligned arrays to .npy without accumulating an episode in RAM."""
import json
from pathlib import Path
import numpy as np

SCALARS = ('action_loss', 'video_loss', 'cosine_action', 'cosine_video',
           'gradnorm_vision_action', 'gradnorm_vision_video')


class EpisodeWriter:
    def __init__(self, destination, frame_count, action_dims, save_meta=True):
        self.root = Path(destination)
        self.root.mkdir(parents=True, exist_ok=False)
        self.frame_count = frame_count
        self.action_dims = list(action_dims)
        self.save_meta = save_meta
        self.arrays = {}
        self.completed = set()

    def _write(self, filename, frame, tensor):
        value = tensor.detach().float().cpu().numpy()
        if filename not in self.arrays:
            target = self.root / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            # Files remain explicitly partial until the whole episode is done.
            self.arrays[filename] = np.lib.format.open_memmap(str(target) + '.partial', mode='w+',
                                                            dtype=np.float32, shape=(self.frame_count, *value.shape))
        self.arrays[filename][frame] = value

    def write(self, frame, result):
        k = result['step']
        for name in SCALARS:
            filename = f'{name}_step_{k}.npy' if name.startswith('gradnorm') else f'{name}_{k}.npy'
            self._write(filename, frame, result[name].reshape(()))
        if self.save_meta:
            for prefix in ('u', 'v'):
                if prefix == 'u' and k != 0:
                    continue
                suffix = '' if prefix == 'u' else f'_{k}'
                self._write(f'meta/{prefix}_action{suffix}.npy', frame,
                            result[f'{prefix}_action'][0, :, self.action_dims])
                self._write(f'meta/{prefix}_video{suffix}.npy', frame, result[f'{prefix}_video'][0])
        self.completed.add((frame, k))

    def finish(self, steps, metadata):
        if len(self.completed) != self.frame_count * steps:
            raise ValueError('Cannot finalize an incomplete episode')
        for name, array in self.arrays.items():
            array.flush()
            (self.root / (name + '.partial')).rename(self.root / name)
        (self.root / 'episode.json').write_text(json.dumps(metadata, indent=2) + '\n')
        (self.root / 'COMPLETE').touch()
        self.arrays.clear()
