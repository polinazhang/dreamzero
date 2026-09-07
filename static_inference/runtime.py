"""Load a checkpoint once using the original policy and transform configuration."""
import os
from pathlib import Path


def local_tokenizer():
    """Use the original UMT5 tokenizer already cached for offline inference."""
    from huggingface_hub import snapshot_download
    return snapshot_download('google/umt5-xxl', local_files_only=True)


def load_policy(checkpoint, embodiment):
    import torch
    # Match socket_test_optimized_AR.main before importing/constructing the policy.
    # Its compiled autoregressive scheduler needs more than Dynamo's default 8 variants.
    os.environ['ATTENTION_BACKEND'] = 'TE'
    torch._dynamo.config.recompile_limit = 800
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    if not dist.is_initialized():
        dist.init_process_group('nccl')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    if dist.get_world_size() != 1:
        raise ValueError('Use one GPU/process per shared model; original inference P2P breaks autograd')
    mesh = init_device_mesh('cuda', (1,), mesh_dim_names=('ip',))
    return GrootSimPolicy(embodiment_tag=EmbodimentTag(embodiment), model_path=str(Path(checkpoint)),
                          device='cuda', device_mesh=mesh, tokenizer_path_override=local_tokenizer())
