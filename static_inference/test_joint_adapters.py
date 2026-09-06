"""Per-combination rule tests using original checkpoint statistics on CPU."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from .check_joint_adapters import cpu_policy
from .droid import DroidAdapter
from .yam import YamAdapter
from .openarm import OpenArmAdapter


class FixtureEpisode:
    """Synthetic images are TEST ONLY; production readers require every camera."""
    def __init__(self,width,cameras):
        self.state=np.tile(np.linspace(.01,.16,width,dtype=np.float32),(60,1))
        self.actions=self.state+np.linspace(.001,.016,width,dtype=np.float32)
        self.camera_shapes={camera:(64,64) for camera in cameras}
    def __len__(self):return len(self.state)
    def images(self,camera,indices):return np.zeros((len(indices),64,64,3),dtype=np.uint8)
    def instruction(self,t):return 'Move the object.'


class JointAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]/'checkpoints'
        cls.droid=cpu_policy(root/'DreamZero-DROID','oxe_droid')
        cls.agibot=cpu_policy(root/'DreamZero-AgiBot','agibot')

    def verify(self,adapter,expected_slots):
        policy=adapter.policy
        episode=FixtureEpisode(len(expected_slots),[v for _,v in adapter.mapping.cameras])
        sample=adapter.sample(episode,0)
        self.assertEqual(sample.inputs.state_mask[0,0].nonzero().flatten().tolist(),sorted(expected_slots))
        self.assertEqual(sample.action_mask[0,0].nonzero().flatten().tolist(),sorted(expected_slots))
        self.assertEqual(sample.action_mask.sum().item(),len(expected_slots)*adapter.horizon)
        self.assertEqual(sample.inputs.state[~sample.inputs.state_mask].count_nonzero().item(),0)
        self.assertEqual(sample.action[~sample.action_mask].count_nonzero().item(),0)
        # Independently assemble expected native coordinates and apply ORIGINAL
        # per-field normalizers; invalid coordinates must be zero afterward.
        source_to_slot=dict(enumerate(expected_slots))
        reverse={slot:source for source,slot in source_to_slot.items()}
        relative=set(policy.train_cfg.relative_action_keys)
        for modality,groups,output in [('state',adapter.state_groups,sample.inputs.state[0]),
                                      ('action',adapter.action_groups,sample.action[0])]:
            for key,start,end in groups:
                transform=next(t for t in policy.eval_transform.transforms
                               if hasattr(t,'_normalizers') and key in t._normalizers)
                raw=np.zeros((1 if modality=='state' else adapter.horizon,end-start),dtype=np.float32)
                for slot in range(start,end):
                    if slot in reverse:
                        src=reverse[slot]
                        raw[:,slot-start]=episode.state[0,src] if modality=='state' else episode.actions[:adapter.horizon,src]
                        if modality=='action' and key.split('.',1)[1] in relative:
                            raw[:,slot-start]-=episode.state[0,src]
                normalized=transform._normalizers[key].forward(torch.from_numpy(raw)).float()
                for slot in range(start,end):
                    if slot in reverse:
                        torch.testing.assert_close(output[:,slot].float(),normalized[:,slot-start])
        return sample

    def test_droid_all_native_joint_positions_and_gripper(self):
        self.verify(DroidAdapter(self.droid),list(range(8)))

    def test_yam_both_six_joint_arms_and_grippers(self):
        sample=self.verify(YamAdapter(self.agibot),[0,1,2,3,4,5,14,7,8,9,10,11,12,15])
        self.assertEqual(sample.inputs.state[...,6].count_nonzero().item(),0)
        self.assertEqual(sample.inputs.state[...,13].count_nonzero().item(),0)
        self.assertFalse(sample.action_mask[...,6].any())
        self.assertFalse(sample.action_mask[...,13].any())

    def test_openarm_both_seven_joint_arms_and_grippers(self):
        self.verify(OpenArmAdapter(self.agibot),list(range(16)))

    def test_wrong_checkpoint_embodiment_rejected(self):
        with self.assertRaises(ValueError):YamAdapter(self.droid)
        with self.assertRaises(ValueError):OpenArmAdapter(self.droid)
        with self.assertRaises(ValueError):DroidAdapter(self.agibot)


if __name__=='__main__':unittest.main()
