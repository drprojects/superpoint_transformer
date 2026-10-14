import pytest
import torch

from src.data import Data
from src.transforms import CutPursuitPartition
from src.utils.ground import single_plane_model


def test_data_select_keeps_empty_edge_index():
    """An empty `edge_index` must stay consistent with `edge_attr`."""
    data = Data(
        pos=torch.rand(4, 3),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 7))

    out = data.select(torch.tensor([0, 2]))[0]

    assert out.edge_index is not None
    assert out.edge_index.shape == (2, 0)
    assert out.edge_attr.shape == (0, 7)


def test_to_trimmed_without_edges():
    data = Data(pos=torch.rand(3, 3))
    assert data.to_trimmed() is data


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize('pos', [
    torch.zeros(1, 3),
    torch.ones(5, 3),
    torch.rand(3, 3)])
@pytest.mark.parametrize('empty_edges', [True, False])
def test_connect_isolated_degenerate(pos, empty_edges):
    """A single node, nodes all sharing the same position, or less
    nodes than `k`, with an empty or missing `edge_index`."""
    data = Data(pos=pos)
    if empty_edges:
        data.edge_index = torch.zeros(2, 0, dtype=torch.long)
    data = data.connect_isolated(k=5)
    if pos.shape[0] > 1:
        assert not data.is_isolated().any()
        assert data.edge_index.ge(0).all()


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"))])
@pytest.mark.parametrize('n', [1, 2])
def test_single_plane_model_too_few_points(device, n):
    """Too few ground candidates to fit a plane: fall back to the
    horizontal plane through the lowest point."""
    pos = torch.tensor([[0., 0., 1.], [1., 0., 2.]], device=device)[:n]
    model = single_plane_model(pos)
    query = torch.tensor([[5., 5., 4.]], device=device)
    assert torch.allclose(model(query).float().cpu(), torch.tensor([3.]))


def _partition_input(pos, edge_index):
    n = pos.shape[0]
    return Data(
        pos=pos,
        x=torch.rand(n, 2),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.shape[1]),
        y=torch.ones(n, 3, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize('edge_index', [
    torch.zeros(2, 0, dtype=torch.long),  # no edge
    torch.tensor([[0, 1], [1, 2]]),       # exactly 2 edges
    torch.tensor([[0, 1, 2], [1, 2, 3]])])
def test_cut_pursuit_partition_degenerate_graphs(edge_index):
    """The partition must output as many levels as requested, even when
    a level cannot be partitioned further."""
    pos = torch.rand(4, 3)
    if edge_index.shape[1] == 0:
        pos, edge_index = pos[:1], edge_index
    data = _partition_input(pos, edge_index)
    partition = CutPursuitPartition(
        regularization=[0.1, 0.2, 0.3], cutoff=1, parallel=False)

    nag = partition(data)

    assert nag.num_levels == 4
    for i in range(1, 4):
        assert nag[i].num_nodes >= 1
