"""OpenArm + DreamZero-AgiBot: both seven-joint arms and both grippers."""
from .joint_adapter import JointMapping, JointDatasetAdapter
from .lerobot_v3 import JointEpisode, inventory

DATASETS=('pick_cup','pour_ice','use_spoon','use_steel_spoon')
MAPPING=JointMapping('openarm','agibot',tuple(range(16)),tuple(range(16)),
                    (('video.top_head','head_image'),('video.hand_left','left_wrist_image'),
                     ('video.hand_right','right_wrist_image')),
                    'annotation.detailed_global_instruction_concise')


class OpenArmAdapter(JointDatasetAdapter):
    def __init__(self,policy):
        super().__init__(policy,MAPPING)


class OpenArmEpisode(JointEpisode):
    def __init__(self,root,entry):
        super().__init__(root,entry,'state','actions',[v for _,v in MAPPING.cameras],16)
