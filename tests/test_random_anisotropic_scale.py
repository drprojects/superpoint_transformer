from unittest import mock

import pytest
import torch

from src.data import NAG, Data
from src.transforms import RandomAnisotropicScale
from src.transforms.graph import _minimalistic_horizontal_edge_features


def _toy_superedges(seed=0):
    """Level-1 data with 3 trimmed superedges, and the level-0 point
    pairs ('subedges') making up each of them.
    """
    g = torch.Generator().manual_seed(seed)
    # Small extent so that `std_off` stays below its clip value of 2
    points = torch.rand(60, 3, generator=g) * 3
    se_point_index = torch.randint(0, 60, (2, 30), generator=g)
    se_id = torch.arange(30) % 3
    data = Data(
        pos=torch.rand(4, 3, generator=g) * 10,
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]))
    return data, points, se_point_index, se_id


def _edge_attr_7(points, seed=0):
    data, _, se_point_index, se_id = _toy_superedges(seed)
    data = _minimalistic_horizontal_edge_features(
        data, points, se_point_index, se_id)
    return data.edge_attr.float()


@pytest.mark.parametrize('s', [0.8, 1.0, 1.2])
def test_anisotropic_scale_matches_edge_features_of_scaled_points(s):
    """Scaling the preprocessed 7-column `edge_attr` with
    `RandomAnisotropicScale` must give the features that would have
    been computed on the scaled points. It used to multiply `std_off`
    and `mean_dist` by `scale.norm()` (~sqrt(3) * s), inflating them at
    train time only.
    """
    _, points, _, _ = _toy_superedges()
    expected = _edge_attr_7(points * s)

    d0 = Data(
        pos=points.clone(),
        super_index=torch.arange(60) % 4)
    d1 = Data(
        pos=torch.rand(4, 3),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
        edge_attr=_edge_attr_7(points))
    nag = NAG([d0, d1])

    # delta = 0.2 and rand = u give a scale of 1 + 0.4 * u - 0.2
    u = (s - 0.8) / 0.4
    with mock.patch(
            'src.transforms.geometry.torch.rand',
            return_value=torch.tensor([[u]])):
        nag = RandomAnisotropicScale(delta=0.2, per_axis=False)(nag)

    torch.testing.assert_close(
        nag[1].edge_attr, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(nag[0].pos, points * s)


def _scale_factors(per_axis, seed):
    """Per-axis factors applied to the positions by one call of
    `RandomAnisotropicScale(delta=0.2)`.
    """
    pos = torch.ones(5, 3)
    nag = NAG([Data(pos=pos.clone())])
    torch.manual_seed(seed)
    nag = RandomAnisotropicScale(delta=0.2, per_axis=per_axis)(nag)
    return nag[0].pos[0] / pos[0]


@pytest.mark.parametrize('seed', range(5))
def test_anisotropic_scale_draws_one_factor_per_axis(seed):
    """By default, each axis must have its own scaling factor. A single
    random draw used to be shared by the 3 axes, so that the
    transform was isotropic despite its name.
    """
    scale = _scale_factors(per_axis=True, seed=seed)
    assert ((0.8 <= scale) & (scale <= 1.2)).all()
    assert not torch.allclose(scale, scale[0].expand(3))


@pytest.mark.parametrize('seed', range(5))
def test_anisotropic_scale_shared_factor_if_not_per_axis(seed):
    scale = _scale_factors(per_axis=False, seed=seed)
    assert ((0.8 <= scale) & (scale <= 1.2)).all()
    torch.testing.assert_close(scale, scale[0].expand(3))


def test_anisotropic_scale_keeps_normals_orthogonal_to_surface():
    """After an anisotropic scaling, normals must stay orthogonal to
    the scaled surface. They used to be multiplied by the scale instead
    of divided by it.
    """
    g = torch.Generator().manual_seed(0)
    normal = torch.nn.functional.normalize(
        torch.randn(8, 3, generator=g), dim=1)
    # Two tangent vectors per normal
    tangent_1 = torch.nn.functional.normalize(
        torch.cross(normal, torch.randn(8, 3, generator=g), dim=1), dim=1)
    tangent_2 = torch.cross(normal, tangent_1, dim=1)
    nag = NAG([Data(pos=torch.rand(8, 3, generator=g), normal=normal)])

    # delta = 0.2 and rand = u give scales of 1 + 0.4 * u - 0.2
    u = torch.tensor([[0., 0.5, 1.]])
    scale = 1 + 0.4 * u - 0.2
    with mock.patch(
            'src.transforms.geometry.torch.rand', return_value=u):
        nag = RandomAnisotropicScale(delta=0.2)(nag)

    new_normal = nag[0].normal
    torch.testing.assert_close(new_normal.norm(dim=1), torch.ones(8))
    for tangent in [tangent_1, tangent_2]:
        torch.testing.assert_close(
            (new_normal * tangent * scale).sum(dim=1), torch.zeros(8),
            rtol=0, atol=1e-6)
