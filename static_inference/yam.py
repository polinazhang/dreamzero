"""MolmoAct2-YAM + DreamZero-AgiBot: both six-joint arms and both grippers."""
from .joint_adapter import JointMapping, JointDatasetAdapter
from .lerobot_v3 import JointEpisode, inventory

# Source order: L0..L5, LG, R0..R5, RG. Do not shift the right arm into
# checkpoint slot 6: that is the unused seventh LEFT joint.
SLOTS=(0,1,2,3,4,5,14,7,8,9,10,11,12,15)
MAPPING=JointMapping('yam','agibot',SLOTS,SLOTS,
                    (('video.top_head','observation.images.top'),
                     ('video.hand_left','observation.images.left'),
                     ('video.hand_right','observation.images.right')),
                    'annotation.detailed_global_instruction_concise')


class YamAdapter(JointDatasetAdapter):
    def __init__(self,policy):
        super().__init__(policy,MAPPING)


class YamEpisode(JointEpisode):
    def __init__(self,root,entry):
        super().__init__(root,entry,'observation.state','action',[v for _,v in MAPPING.cameras],14)
