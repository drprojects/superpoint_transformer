import warnings
from contextlib import contextmanager

import pytest
import torch

from src.data import Data, NAG, NAGBatch, Cluster, ClusterBatch
from src.transforms import (
    NAGCast,
    NodeSize,
    NAGJitterKey,
    RandomTiltAndRotate,
    RandomAxisFlip,
    NAGDropoutColumns,
    NAGDropoutRows,
    NAGColorAutoContrast,
    NAGColorDrop,
    OnTheFlyHorizontalEdgeFeatures,
    NAGAddSelfLoops)
from src.utils import multinomial_no_replacement, sparse_sample


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")


@contextmanager
def record_cpu_gpu_syncs():
    """Record the CPU-GPU syncs, including blocking host-to-device
    copies, triggered by the operations run in the context. The
    returned list is filled with the corresponding warnings when
    exiting the context.

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


def _format_syncs(syncs):
    return '\n'.join(f'{w.filename}:{w.lineno}' for w in syncs)


def _trimmed_edges(num_nodes, num_edges, g):
    i, j = torch.triu_indices(num_nodes, num_nodes, offset=1)
    perm = torch.randperm(i.shape[0], generator=g)[:num_edges]
    return torch.stack((i[perm], j[perm]))


def _segment_level(num_nodes, num_edges, sub_super_index, g):
    """A level of segments, with the attributes expected by the
    on-device transforms of the EZ-SP configs."""
    data = Data(
        pos=torch.rand(num_nodes, 3, generator=g),
        normal=torch.nn.functional.normalize(
            torch.randn(num_nodes, 3, generator=g), dim=1),
        edge_index=_trimmed_edges(num_nodes, num_edges, g),
        edge_attr=torch.randn(num_edges, 7, generator=g),
        sub=Cluster(
            sub_super_index,
            torch.arange(sub_super_index.shape[0]),
            dense=True))
    for key in ['log_length', 'log_surface', 'log_volume', 'log_size']:
        data[key] = torch.randn(num_nodes, 1, generator=g)
    return data


def _synthetic_nag(seed, num_nodes=(200, 40, 8)):
    g = torch.Generator().manual_seed(seed)
    n0, n1, n2 = num_nodes
    super_index_0 = torch.arange(n0) % n1
    super_index_1 = torch.arange(n1) % n2
    data_0 = Data(
        pos=torch.rand(n0, 3, generator=g),
        intensity=torch.rand(n0, 1, generator=g),
        rgb=torch.rand(n0, 3, generator=g),
        super_index=super_index_0)
    data_1 = _segment_level(n1, 3 * n1, super_index_0, g)
    data_1.super_index = super_index_1
    data_2 = _segment_level(n2, 2 * n2, super_index_1, g)
    return NAG([data_0, data_1, data_2])


def _synthetic_batch():
    """Batch of NAGs on the GPU, as passed to the on-device
    transforms."""
    nag_list = [_synthetic_nag(seed).cuda() for seed in range(2)]
    return NAGBatch.from_nag_list(nag_list)


EZSP_EDGE_HF = [
    'mean_off', 'std_off', 'mean_dist', 'angle_source', 'angle_target',
    'centroid_dir', 'centroid_dist', 'normal_angle', 'log_length',
    'log_surface', 'log_volume', 'log_size']


# On-device transforms expected to run without any CPU-GPU sync. The
# sampling transforms (e.g. `SampleSubNodes`, `SampleSegments`) are not
# listed here: the size of their output depends on the data, which
# cannot be known on CPU without a sync
SYNC_FREE_TRANSFORMS = [
    NAGCast(),
    NodeSize(),
    NAGJitterKey(key='pos', sigma=0.01, trunc=0.05),
    NAGJitterKey(key='edge_attr', sigma=0.1, trunc=0.2),
    RandomTiltAndRotate(phi=0.1, theta=180),
    RandomAxisFlip(p=0.0),
    RandomAxisFlip(p=1.0),
    NAGDropoutColumns(p=0.3, key='edge_attr', inplace=True),
    NAGDropoutColumns(p=0.3, key='edge_attr', inplace=True, to_mean=True),
    NAGDropoutRows(p=0.3, key='edge_attr'),
    NAGDropoutRows(p=0.3, key='edge_attr', to_mean=True),
    NAGColorAutoContrast(p=1.0),
    NAGColorDrop(p=1.0),
    OnTheFlyHorizontalEdgeFeatures(keys=EZSP_EDGE_HF),
    NAGAddSelfLoops()]


@pytest.mark.parametrize(
    'transform', SYNC_FREE_TRANSFORMS, ids=lambda t: repr(t))
def test_on_device_transform_has_no_cpu_gpu_sync(transform):
    nag = _synthetic_batch()
    torch.cuda.synchronize()

    with record_cpu_gpu_syncs() as syncs:
        transform(nag)

    assert len(syncs) == 0, \
        f"{transform} caused {len(syncs)} CPU-GPU sync(s):\n" \
        f"{_format_syncs(syncs)}"


def test_record_cpu_gpu_syncs_detects_syncs():
    """Make sure the recording mechanism actually catches syncs,
    including blocking host-to-device copies."""
    x = torch.ones(3, device='cuda')
    with record_cpu_gpu_syncs() as syncs:
        x.sum().item()
        torch.tensor([1., 2.], device='cuda')
    assert len(syncs) == 2


@pytest.mark.parametrize('num_samples', [1, 7, 100])
def test_multinomial_no_replacement(num_samples):
    """Same output and random generator state as `torch.multinomial`,
    without CPU-GPU sync."""
    weights = torch.rand(100, device='cuda')
    weights[::3] = 0
    num_samples = min(num_samples, int(weights.gt(0).sum()))
    for seed in range(5):
        torch.manual_seed(seed)
        expected = torch.multinomial(weights, num_samples, replacement=False)
        expected_next = torch.rand(3, device='cuda')
        torch.manual_seed(seed)
        torch.cuda.synchronize()
        with record_cpu_gpu_syncs() as syncs:
            out = multinomial_no_replacement(weights, num_samples)
        assert len(syncs) == 0, _format_syncs(syncs)
        assert torch.equal(out, expected)
        assert torch.equal(torch.rand(3, device='cuda'), expected_next)


@pytest.mark.parametrize('use_mask', [False, True])
def test_sparse_sample_num_segments(use_mask):
    """Passing `num_segments` gives the same sampling as before."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 50, (1000,), generator=g).cuda()
    idx[:50] = torch.arange(50, device='cuda')
    mask = torch.rand(1000, generator=g).cuda() > 0.3 if use_mask else None
    for seed in range(3):
        torch.manual_seed(seed)
        expected, expected_ptr = sparse_sample(
            idx, n_max=8, mask=mask, return_pointers=True)
        torch.manual_seed(seed)
        out, ptr = sparse_sample(
            idx, n_max=8, mask=mask, return_pointers=True, num_segments=50)
        assert torch.equal(out, expected)
        assert torch.equal(ptr, expected_ptr)


def test_cluster_from_list_has_no_cpu_gpu_sync():
    """`ClusterBatch.from_list` reads the static `is_index_value` of
    `Cluster` rather than its tensor on the GPU."""
    clusters = [
        Cluster(
            torch.arange(0, 2 * n + 1, 2, device='cuda'),
            torch.randperm(2 * n, device='cuda'))
        for n in (3, 5, 4)]
    torch.cuda.synchronize()
    with record_cpu_gpu_syncs() as syncs:
        batch = ClusterBatch.from_list(clusters)
    assert len(syncs) == 0, _format_syncs(syncs)
    assert torch.equal(
        batch.points.cpu(),
        torch.cat([c.points.cpu() + o for c, o in zip(clusters, (0, 6, 16))]))
