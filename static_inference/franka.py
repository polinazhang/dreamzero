"""Reconstructed Franka absolute joint targets with DreamZero-DROID statistics."""
from pathlib import Path
from .joint_adapter import JointMapping, JointDatasetAdapter
from .lerobot_v3 import JointEpisode

DATA_ROOT = Path('/storage/home/hcoda1/5/xzhang3205/scratch/franka-datasets/sample-reconstructed-joint')
MAPPING = JointMapping('franka', 'oxe_droid', tuple(range(8)), tuple(range(8)),
    (('video.exterior_image_1_left', 'observation.images.camera_front'),
     ('video.exterior_image_2_left', 'observation.images.camera_side'),
     ('video.wrist_image_left', 'observation.images.camera_wrist')),
    'annotation.language.language_instruction')

class FrankaAdapter(JointDatasetAdapter):
    def __init__(self, policy):
        super().__init__(policy, MAPPING)

class FrankaEpisode(JointEpisode):
    def __init__(self, root, entry):
        super().__init__(root, entry, 'observation.state', 'action',
                         [v for _, v in MAPPING.cameras], 8)
