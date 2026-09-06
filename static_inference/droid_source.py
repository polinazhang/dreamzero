"""Read the requested DROID TFDS/TFRecord release without importing TensorFlow."""
import functools
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image


@functools.lru_cache(maxsize=1)
def example_type():
    # TensorFlow Example's public protobuf wire schema. A private descriptor
    # pool avoids any dependency on, or collision with, a TensorFlow runtime.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    file = descriptor_pb2.FileDescriptorProto(name='static_tf_example.proto', package='static_tf', syntax='proto3')
    def message(name):
        return file.message_type.add(name=name)
    def field(owner, name, number, typ, repeated=False, type_name=None):
        f = owner.field.add(name=name, number=number, type=typ, label=3 if repeated else 1)
        if type_name:
            f.type_name = type_name
        return f
    for name, typ in [('BytesList',12), ('FloatList',2), ('Int64List',3)]:
        field(message(name),'value',1,typ,True)
    feature = message('Feature')
    for number, name, target in [(1,'bytes_list','BytesList'),(2,'float_list','FloatList'),(3,'int64_list','Int64List')]:
        field(feature,name,number,11,type_name='.static_tf.'+target)
    features = message('Features')
    entry = features.nested_type.add(name='FeatureEntry')
    entry.options.map_entry = True
    field(entry,'key',1,9)
    field(entry,'value',2,11,type_name='.static_tf.Feature')
    field(features,'feature',1,11,True,'.static_tf.Features.FeatureEntry')
    field(message('Example'),'features',1,11,type_name='.static_tf.Features')
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('static_tf.Example'))


def inventory(root, limit=100):
    root = Path(root)
    info = json.loads((root/'dataset_info.json').read_text())
    if info.get('version') != '1.0.0':
        raise ValueError('Expected the requested DROID 1.0.0 release')
    shards = sorted(root.glob('*-train.tfrecord-*-of-*'))
    expected = next(s for s in info['splits'] if s['name']=='train')['shardLengths']
    if len(shards) != len(expected) or sum(map(int,expected)) != 100:
        raise ValueError('Expected all 31 shards of the requested 100-demonstration release')
    entries = []
    for shard, count in zip(shards, expected):
        with shard.open('rb') as stream:
            size = shard.stat().st_size
            for _ in range(int(count)):
                header = stream.read(12)
                if len(header) != 12:
                    raise ValueError(f'Truncated TFRecord header: {shard}')
                length = struct.unpack('<Q',header[:8])[0]
                offset = stream.tell()
                if offset+length+4 > size:
                    raise ValueError(f'Truncated TFRecord: {shard}')
                entries.append(dict(episode_index=len(entries),shard=str(shard),offset=offset,bytes=length))
                stream.seek(length+4,io.SEEK_CUR)
            if stream.tell() != size:
                raise ValueError(f'Shard record count disagrees with dataset_info: {shard}')
    if not 1 <= limit <= 100:
        raise ValueError('DROID limit must be 1..100')
    return entries[:limit]


class DroidEpisode:
    def __init__(self, entry):
        self.index = entry['episode_index']
        with open(entry['shard'],'rb') as stream:
            stream.seek(entry['offset'])
            payload = stream.read(entry['bytes'])
        if len(payload) != entry['bytes']:
            raise ValueError('Truncated DROID episode')
        self.features = example_type().FromString(payload).features.feature
        self.cameras = ('exterior_image_1_left','exterior_image_2_left','wrist_image_left')
        self.encoded = {k:list(self.features['steps/observation/'+k].bytes_list.value) for k in self.cameras}
        self.length = len(self.encoded[self.cameras[0]])
        if not self.length or any(len(v)!=self.length for v in self.encoded.values()):
            raise ValueError('DROID camera lengths disagree')
        self.state = np.concatenate([self.numeric('steps/observation/joint_position',7),
                                     self.numeric('steps/observation/gripper_position',1)],axis=1)
        # The joint-position action exists: the rule forbids selecting the
        # Cartesian/velocity alternatives when the matching representation exists.
        self.actions = np.concatenate([self.numeric('steps/action_dict/joint_position',7),
                                       self.numeric('steps/action_dict/gripper_position',1)],axis=1)
        self.prompts = [v.decode('utf-8') for v in self.features['steps/language_instruction'].bytes_list.value]
        if len(self.prompts) != self.length:
            raise ValueError('DROID instruction length mismatch')
        self.camera_shapes = {k:self.image(k,0).shape[:2] for k in self.cameras}
        self.source = entry

    def numeric(self,key,width):
        if key not in self.features:
            raise KeyError(f'Missing matching joint/gripper representation: {key}')
        feature = self.features[key]
        if feature.HasField('bytes_list'):
            # TFDS serializes float64 tensors as raw little-endian bytes.
            values = np.frombuffer(b''.join(feature.bytes_list.value),dtype='<f8')
        elif feature.HasField('float_list'):
            values = np.asarray(feature.float_list.value,dtype=np.float32)
        else:
            raise ValueError(f'Unexpected numeric encoding for {key}')
        if values.size != self.length*width:
            raise ValueError(f'{key}: {values.size} values, expected {self.length*width}')
        return values.reshape(self.length,width)

    def __len__(self):
        return self.length

    @functools.lru_cache(maxsize=192)
    def image(self,camera,index):
        with Image.open(io.BytesIO(self.encoded[camera][int(index)])) as image:
            return np.asarray(image.convert('RGB'))

    def images(self,camera,indices):
        return np.stack([self.image(camera,int(i)) for i in indices])

    def instruction(self,t):
        return self.prompts[t]

    def close(self):
        self.image.cache_clear()
