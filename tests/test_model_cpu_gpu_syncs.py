import warnings
from contextlib import contextmanager

import pytest
import torch
from torch_scatter import scatter_sum

from src.nn import UnitSphereNorm
from src.utils.nn import build_qk_scale_func
from src.utils.scatter import scatter_mean_weighted


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")


@contextmanager
def record_cpu_gpu_syncs():
    """Record the CPU-GPU syncs triggered by the operations run in the
    context. The returned list is filled when exiting the context.

    NB: this relies on `torch.cuda.set_sync_debug_mode`, which does not
    detect all synchronizing operations.
    """
    syncs = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.cuda.set_sync_debug_mode('warn')
        try:
            yield syncs
        finally:
            torch.cuda.set_sync_debug_mode('default')
    syncs.extend(
        w for w in caught
        if 'called a synchronizing CUDA operation' in str(w.message))


@pytest.mark.parametrize('qk_scale', [None, 'd+g', 'dg', 'g'])
def test_qk_scale_num_nodes(qk_scale):
    """Passing the number of nodes gives the same qk-scale as
    `bincount`, without CPU-GPU sync, even when the last nodes have no
    edge."""
    f = build_qk_scale_func(64, 4, qk_scale)
    num_nodes = 100
    g = torch.Generator().manual_seed(0)
    s = torch.randint(0, num_nodes - 10, (500,), generator=g).cuda()
    expected = f(s)
    torch.cuda.synchronize()

    with record_cpu_gpu_syncs() as syncs:
        out = f(s, num_nodes=num_nodes)

    assert len(syncs) == 0
    assert torch.equal(out, expected)


def test_scatter_mean_weighted_no_sync():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 3, generator=g)
    idx = torch.randint(0, 100, (1000,), generator=g)
    w = torch.randint(0, 3, (1000,), generator=g)  # zero-weight segments

    # Reference: segment means with weights, 0 for empty segments
    w_sum = scatter_sum(w.float(), idx, dim_size=120)
    expected = scatter_sum(x * w.view(-1, 1), idx, dim=0, dim_size=120) \
        / w_sum.clamp(min=1).view(-1, 1)
    torch.testing.assert_close(
        scatter_mean_weighted(x, idx, w, dim_size=120), expected)

    x, idx, w = x.cuda(), idx.cuda(), w.cuda()
    torch.cuda.synchronize()
    with record_cpu_gpu_syncs() as syncs:
        scatter_mean_weighted(x, idx, w, dim_size=120)
    assert len(syncs) == 0


def test_unit_sphere_norm_weighted_no_sync():
    pos = torch.rand(50, 3, device='cuda')
    w = torch.zeros(50, device='cuda')
    torch.cuda.synchronize()
    with record_cpu_gpu_syncs() as syncs:
        out, _ = UnitSphereNorm()(pos, None, w=w)
    assert len(syncs) == 0
    # With all-zero weights, the center falls back to the origin
    torch.testing.assert_close(
        out, pos / ((pos.max(0).values - pos.min(0).values).max() + 1e-2))
