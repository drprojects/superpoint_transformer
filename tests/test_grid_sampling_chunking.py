import pytest
import torch
from torch_scatter import scatter_mean

from src.data import Data, InstanceData
from src.transforms import GridSampling3D, SaveNodeIndex


@pytest.mark.parametrize('chunk_size', [None, 60])
def test_grid_sampling_chunking_keeps_sub_indices(chunk_size):
    """`sub` points must stay indices into the input cloud when
    GridSampling3D processes it in chunks."""
    g = torch.Generator().manual_seed(0)
    n = 200
    data = Data(
        pos=torch.rand(n, 3, generator=g) * 5,
        v=torch.arange(n, dtype=torch.float))
    data = SaveNodeIndex(key='sub')(data)
    out = GridSampling3D(size=1.0, chunk_size=chunk_size)(data.clone())

    points = out.sub.points
    assert int(points.max()) < n
    assert points.unique().numel() == n
    voxel_of_point = torch.arange(
        out.num_nodes).repeat_interleave(out.sub.sizes.long())
    assert torch.allclose(
        scatter_mean(data.v[points], voxel_of_point, dim=0), out.v)


def test_grid_sampling_chunking_keeps_shared_indices():
    """Chunking must not change index-valued attributes whose values
    are shared by all chunks: objects, superpoints, input clouds."""
    g = torch.Generator().manual_seed(0)
    n = 2000
    pos = torch.rand(n, 3, generator=g) * 10
    y = torch.randint(0, 5, (n,), generator=g)
    obj = torch.randint(0, 30, (n,), generator=g)
    ones = torch.ones_like(obj)
    data = Data(
        pos=pos,
        y=y,
        obj=InstanceData(torch.arange(n), obj, ones, y, dense=True),
        obj_pred=InstanceData(
            torch.arange(n), (obj + 7) % 30, ones, y, dense=True),
        super_index=(pos[:, 0] // 2).long() * 5 + (pos[:, 1] // 2).long(),
        batch=(pos[:, 2] > 5).long())
    data = SaveNodeIndex(key='sub')(data)

    ref = GridSampling3D(size=0.5, chunk_size=None)(data.clone())
    out = GridSampling3D(size=0.5, chunk_size=300)(data.clone())

    # Chunks are made of consecutive voxels, so the voxel order is kept
    assert torch.equal(ref.pos, out.pos)
    for key in ['super_index', 'batch']:
        assert torch.equal(ref[key], out[key]), key
    for key in ['obj', 'obj_pred']:
        assert torch.equal(ref[key].indices, out[key].indices), key
        assert torch.equal(ref[key].obj, out[key].obj), key
        assert torch.equal(ref[key].count, out[key].count), key

    # Points may be ordered differently inside each voxel
    def sorted_sub(sub):
        voxel = torch.arange(sub.num_groups).repeat_interleave(sub.sizes)
        return (voxel * n + sub.points).sort().values

    assert torch.equal(ref.sub.pointers, out.sub.pointers)
    assert torch.equal(sorted_sub(ref.sub), sorted_sub(out.sub))
