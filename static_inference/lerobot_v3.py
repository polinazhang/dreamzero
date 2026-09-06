"""LeRobot v3 episode/file addressing and timestamp-aligned bounded video reads."""
from collections import OrderedDict
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def inventory(root,limit):
    root=Path(root)
    info=json.loads((root/'meta/info.json').read_text())
    if info.get('codebase_version')!='v3.0':
        raise ValueError(f'Expected LeRobot v3: {root}')
    if limit<1:
        raise ValueError('Episode limit must be positive')
    paths=sorted((root/'meta/episodes').rglob('*.parquet'))
    if not paths:
        raise FileNotFoundError(root/'meta/episodes')
    columns=[k for k in pq.read_schema(paths[0]).names if not k.startswith('stats/')]
    # Keep all episode addresses before selecting, rather than confusing
    # metadata-file ordering with episode ordering.
    entries=pq.read_table(paths,columns=columns).to_pylist()
    entries=sorted(entries,key=lambda row:row['episode_index'])
    ids=[row['episode_index'] for row in entries]
    if len(set(ids))!=len(ids) or len(entries)<limit:
        raise ValueError(f'Insufficient or duplicate episodes in {root}: requested {limit}, found {len(entries)}')
    selected=entries[:limit]
    cameras=[k for k,v in info['features'].items() if v['dtype']=='video']
    for entry in selected:
        for path in addressed_files(root,info,entry,cameras):
            if not path.is_file():
                raise FileNotFoundError(path)
    return selected


def addressed_files(root,info,entry,cameras):
    root=Path(root)
    yield root/info['data_path'].format(chunk_index=entry['data/chunk_index'],file_index=entry['data/file_index'])
    for camera in cameras:
        prefix='videos/'+camera+'/'
        yield root/info['video_path'].format(video_key=camera,chunk_index=entry[prefix+'chunk_index'],
                                            file_index=entry[prefix+'file_index'])


class TimestampVideo:
    """Sequential decode with a bounded frame cache and seeks for older requests."""
    def __init__(self,path,timestamps,fps,cache_size=96):
        self.path=Path(path)
        self.timestamps=np.asarray(timestamps,dtype=np.float64)
        self.tolerance=1/fps+1e-4
        self.cache_size=cache_size
        self.cache=OrderedDict()
        self.container=None
        self.previous=self.following=None
        self._seek(self.timestamps[0])

    def _next(self):
        try:
            frame=next(self.decoder)
        except StopIteration:
            return None
        if frame.pts is None:
            raise ValueError(f'Video frame lacks presentation timestamp: {self.path}')
        return float(frame.pts*frame.time_base),frame

    def _seek(self,time):
        if self.container is not None:
            self.container.close()
        self.container=av.open(str(self.path))
        stream=self.container.streams.video[0]
        stream.thread_type='AUTO'
        self.container.seek(max(0,int(time/float(stream.time_base))),stream=stream,backward=True)
        self.decoder=self.container.decode(video=0)
        self.previous=None
        self.following=self._next()
        if self.following is None:
            raise ValueError(f'No video frames at {time}: {self.path}')

    def image(self,index):
        index=int(index)
        if not 0<=index<len(self.timestamps):
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        target=self.timestamps[index]
        if self.previous is not None and target<self.previous[0]-1e-7:
            self._seek(target)
        while self.following is not None and self.following[0]<target:
            self.previous=self.following
            self.following=self._next()
        candidates=[v for v in (self.previous,self.following) if v is not None]
        time,frame=min(candidates,key=lambda value:abs(value[0]-target))
        if abs(time-target)>self.tolerance:
            raise ValueError(f'Video timestamp mismatch at row {index}: wanted {target}, got {time}: {self.path}')
        image=frame.to_ndarray(format='rgb24')
        self.cache[index]=image
        if len(self.cache)>self.cache_size:
            self.cache.popitem(last=False)
        return image

    def images(self,indices):
        return np.stack([self.image(i) for i in indices])

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container=None
        self.cache.clear()


class JointEpisode:
    def __init__(self,root,entry,state_key,action_key,cameras,expected_width):
        self.root=Path(root)
        self.index=int(entry['episode_index'])
        self.source=entry
        self.info=json.loads((self.root/'meta/info.json').read_text())
        data_path=next(addressed_files(self.root,self.info,entry,[]))
        columns=[state_key,action_key,'timestamp','frame_index','episode_index','task_index']
        if 'prompt' in self.info['features']:
            columns.append('prompt')
        table=pq.read_table(data_path,columns=columns,filters=[('episode_index','=',self.index)])
        self.arrays={k:np.asarray(table[k].to_pylist()) for k in columns}
        self.state=np.asarray(self.arrays[state_key],dtype=np.float32)
        self.actions=np.asarray(self.arrays[action_key],dtype=np.float32)
        self.length=int(entry['length'])
        if self.state.shape!=(self.length,expected_width) or self.actions.shape!=self.state.shape:
            raise ValueError(f'Episode shape/metadata mismatch: {data_path}, episode={self.index}')
        if not np.array_equal(self.arrays['frame_index'].reshape(-1),np.arange(self.length)):
            raise ValueError('Episode rows must be in source frame order')
        self.tasks={}
        for row in pq.read_table(self.root/'meta/tasks.parquet').to_pylist():
            text=row.get('task',row.get('__index_level_0__'))
            if text is None:
                raise ValueError('No task text in tasks.parquet')
            self.tasks[int(row['task_index'])]=str(text)
        self.readers={}
        self.camera_shapes={}
        try:
            for camera in cameras:
                prefix='videos/'+camera+'/'
                path=list(addressed_files(self.root,self.info,entry,[camera]))[1]
                timestamps=self.arrays['timestamp'].reshape(-1)+float(entry[prefix+'from_timestamp'])
                if timestamps[-1]>float(entry[prefix+'to_timestamp'])+1/self.info['fps']:
                    raise ValueError('Episode timestamps exceed the recorded video segment')
                self.readers[camera]=TimestampVideo(path,timestamps,self.info['fps'])
                self.camera_shapes[camera]=self.readers[camera].image(0).shape[:2]
        except Exception:
            self.close()
            raise

    def __len__(self):
        return self.length

    def images(self,camera,indices):
        return self.readers[camera].images(indices)

    def instruction(self,t):
        if 'prompt' in self.arrays:
            prompt=np.asarray(self.arrays['prompt'][t]).reshape(-1)[0]
            return str(prompt)
        return self.tasks[int(np.asarray(self.arrays['task_index'][t]).reshape(-1)[0])]

    def close(self):
        for reader in self.readers.values():
            reader.close()
