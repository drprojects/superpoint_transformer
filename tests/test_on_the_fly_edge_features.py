import math

import torch

from src.data import Data
from src.transforms.graph import _on_the_fly_horizontal_edge_features


def test_on_the_fly_angles_follow_edge_direction():
    """For every edge i->j, including the j->i edges appended by
    `_on_the_fly_horizontal_edge_features`, `angle_source` is the angle
    with the normal of i and `angle_target` with the normal of j. The
    j->i edges used to copy the i->j values unchanged.
    """
    g = torch.Generator().manual_seed(0)
    normal = torch.nn.functional.normalize(
        torch.randn(4, 3, generator=g), dim=1)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 3]])
    edge_attr = torch.randn(3, 7, generator=g)
    data = Data(
        pos=torch.rand(4, 3, generator=g),
        normal=normal,
        edge_index=edge_index,
        edge_attr=edge_attr)

    data = _on_the_fly_horizontal_edge_features(
        data,
        keys=['mean_off', 'std_off', 'mean_dist', 'angle_source',
              'angle_target'])

    src, tgt = data.edge_index
    direction = torch.nn.functional.normalize(data.edge_attr[:, :3], dim=1)
    angle_src = (direction * normal[src]).sum(dim=1).abs()
    angle_tgt = (direction * normal[tgt]).sum(dim=1).abs()
    assert data.edge_index.shape[1] == 6
    torch.testing.assert_close(data.edge_attr[:, 7], angle_src)
    torch.testing.assert_close(data.edge_attr[:, 8], angle_tgt)
    assert not math.isclose(
        data.edge_attr[3, 7].item(), data.edge_attr[0, 7].item())
