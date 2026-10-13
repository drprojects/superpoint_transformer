import torch

from src.data import Data, NAG, Cluster


def _level(n, super_index=None, sub_pointers=None, sub_points=None):
    data = Data(pos=torch.rand(n, 3))
    if super_index is not None:
        data.super_index = torch.tensor(super_index)
    if sub_pointers is not None:
        data.sub = Cluster(
            torch.tensor(sub_pointers), torch.tensor(sub_points), dense=False)
    return data


def test_nag_select_keeps_unchanged_upper_levels():
    """Selecting at level 1 while keeping all level-2 parents must not
    drop the `sub` of level 3."""
    nag = NAG([
        _level(8, super_index=[0, 0, 1, 1, 2, 2, 3, 3]),
        _level(4, [0, 0, 1, 1], [0, 2, 4, 6, 8], list(range(8))),
        _level(2, [0, 0], [0, 2, 4], [0, 1, 2, 3]),
        _level(1, None, [0, 2], [0, 1])])

    out = nag.select(1, torch.tensor([0, 1, 2]))

    assert [out[i].num_nodes for i in range(4)] == [6, 3, 2, 1]
    for i in range(3):
        assert out[i].super_index is not None
    for i in range(1, 4):
        assert out[i].sub is not None
    assert torch.equal(out[3].sub.points, torch.tensor([0, 1]))
