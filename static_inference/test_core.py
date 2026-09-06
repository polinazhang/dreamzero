"""CPU checks for loss semantics, local gradients, masks and storage alignment."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .core import StaticInferenceCore, cosine, flow_losses, joint_norm
from .storage import EpisodeWriter


class Weights:
    def training_weight(self, times):
        return 0.25 + times.float() / 1000


class CoreTests(unittest.TestCase):
    def test_original_padded_reduction_and_weights(self):
        ap = torch.tensor([[[1., 2., 30., 40.], [2., 4., 50., 60.]]], requires_grad=True)
        ua = torch.zeros_like(ap)
        vp = torch.arange(8.).reshape(1, 1, 2, 2, 2).requires_grad_()
        uv = torch.zeros((1, 1, 2, 3, 3))  # Original spatial cropping
        mask = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0]]])
        at = torch.tensor([[1000, 500]])
        vt = torch.tensor([[1000, 500]])
        la, lv = flow_losses(ap, vp, ua, uv, mask, torch.tensor([True]), Weights(), at, vt)
        expected_a = (((1 + 4) / 4) * 1.25 + ((4 + 16) / 4) * .75) / 2
        expected_v = (torch.arange(4.).square().mean() * 1.25 + torch.arange(4., 8.).square().mean() * .75) / 2
        torch.testing.assert_close(la, torch.tensor(expected_a))
        torch.testing.assert_close(lv, expected_v)
        grad = torch.autograd.grad(la, ap)[0]
        self.assertEqual(grad[..., 2:].count_nonzero().item(), 0)
        la0, _ = flow_losses(ap, vp, ua, uv, mask, torch.tensor([False]), Weights(), at, vt)
        self.assertEqual(la0.item(), 0)

    def test_joint_local_gradient(self):
        clip = torch.tensor([1., 2.], requires_grad=True)
        vae = torch.tensor([3.], requires_grad=True)
        fixed_latent = torch.tensor([4.]).detach()
        loss = (clip.sum() + 2 * vae.sum() + fixed_latent.sum()).square()
        grads = torch.autograd.grad(loss, (clip, vae))
        expected = torch.tensor(26 * (6 ** .5))
        torch.testing.assert_close(joint_norm(grads), expected)

    def test_cosine_masks_entire_field(self):
        a = torch.tensor([[[1., 99.], [2., -99.]]])
        b = torch.tensor([[[2., -999.], [4., 999.]]])
        mask = torch.tensor([[[1, 0], [1, 0]]])
        torch.testing.assert_close(cosine(a, b, mask), torch.ones(1))

    def test_step_validation(self):
        core = StaticInferenceCore(SimpleNamespace(num_inference_steps=16))
        for value in [0, -1, 17, 1.5]:
            with self.assertRaises(ValueError):
                next(core.evaluate(None, value))

    @unittest.skipUnless(torch.cuda.is_available(), "Original UniPC constructor requires CUDA")
    def test_all_supported_step_counts_integrate_constant_flow(self):
        for decoupled in (False, True):
            head = SimpleNamespace(num_inference_steps=16, sigma_shift=5.0,
                                   scheduler=SimpleNamespace(num_train_timesteps=1000),
                                   config=SimpleNamespace(decouple_inference_noise=decoupled,
                                                          video_inference_final_noise=.8))
            core = StaticInferenceCore(head)
            for count in range(1, 17):
                video, action = core.schedulers(count, torch.device('cuda'))
                for scheduler in (video, action):
                    expected = 1 - .25 * (scheduler.sigmas[0] - scheduler.sigmas[-1]).item()
                    sample = torch.ones(1, 2, 3, device='cuda')
                    for k, time in enumerate(scheduler.timesteps):
                        sample = scheduler.step(torch.full_like(sample, .25), time, sample,
                                                step_index=k, return_dict=False)[0]
                        self.assertTrue(torch.isfinite(sample).all())
                    torch.testing.assert_close(sample, torch.full_like(sample, expected), atol=2e-5, rtol=2e-5)

    def test_storage_alignment_and_meta_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for save_meta in (True, False):
                root = Path(tmp) / str(save_meta)
                writer = EpisodeWriter(root, 2, [0, 2], save_meta)
                for frame in range(2):
                    for step in range(2):
                        result = {name: torch.tensor(float(10*frame+step)) for name in
                                  ('action_loss','video_loss','cosine_action','cosine_video',
                                   'gradnorm_vision_action','gradnorm_vision_video')}
                        result.update(step=step, u_action=torch.arange(12.).reshape(1,3,4),
                                      v_action=torch.arange(12.).reshape(1,3,4),
                                      u_video=torch.zeros(1,2,2,3,3), v_video=torch.zeros(1,2,2,3,3))
                        writer.write(frame, result)
                writer.finish(2, {})
                np.testing.assert_array_equal(np.load(root/'action_loss_1.npy'), [1,11])
                self.assertEqual((root/'meta').exists(), save_meta)
                self.assertFalse(list(root.rglob('*.partial')))
                if save_meta:
                    actual = np.load(root/'meta/u_action.npy')
                    self.assertEqual(actual.shape, (2,3,2))
                    np.testing.assert_array_equal(actual[0], [[0,2],[4,6],[8,10]])


if __name__ == '__main__':
    unittest.main()
