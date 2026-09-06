"""DROID 1.0.0 + DreamZero-DROID: preserve all eight native coordinates."""
from .joint_adapter import JointMapping, JointDatasetAdapter
from .droid_source import DroidEpisode, inventory

MAPPING = JointMapping('droid','oxe_droid',tuple(range(8)),tuple(range(8)),
                       (('video.exterior_image_1_left','exterior_image_1_left'),
                        ('video.exterior_image_2_left','exterior_image_2_left'),
                        ('video.wrist_image_left','wrist_image_left')),
                       'annotation.language.language_instruction')


class DroidAdapter(JointDatasetAdapter):
    def __init__(self,policy):
        super().__init__(policy,MAPPING)
