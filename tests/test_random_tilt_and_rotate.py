import torch

from src.data import NAG, Data
from src.transforms import RandomTiltAndRotate


def _nag(seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(20, 3, generator=g) * 10
    return NAG([
        Data(pos=pos, super_index=torch.arange(20) // 5),
        Data(pos=torch.rand(4, 3, generator=g) * 10)])


def test_rotation_without_tilt():
    """With `phi=0` (no tilt), the nodes must still be rotated around
    the z axis by an angle in [-theta, theta]. `phi=0` used to skip the
    whole rotation.
    """
    torch.manual_seed(0)
    nag = _nag()
    pos = nag[0].pos.clone()

    out = RandomTiltAndRotate(phi=0, theta=180)(nag.clone())

    assert not torch.allclose(out[0].pos, pos)
    torch.testing.assert_close(out[0].pos[:, 2], pos[:, 2])
    torch.testing.assert_close(
        out[0].pos[:, :2].norm(dim=1), pos[:, :2].norm(dim=1))


def test_no_tilt_no_rotation_is_identity():
    nag = _nag()
    pos = nag[0].pos.clone()

    out = RandomTiltAndRotate(phi=0, theta=0)(nag.clone())

    torch.testing.assert_close(out[0].pos, pos)


def _rotation_axis_tilt(pos_in, pos_out):
    """Angle (degrees) between the z axis and the axis of the rotation
    mapping `pos_in` to `pos_out`.
    """
    R = torch.linalg.lstsq(pos_in, pos_out).solution.T
    axis = torch.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    axis = axis / axis.norm()
    return torch.rad2deg(torch.acos(axis[2].abs()))


def test_tilt_amplitude_matches_phi():
    """`phi` is documented as 3 standard deviations of the tilt. The xy
    offset of the axis used to be drawn with a covariance `sigma I`,
    i.e. a standard deviation `sqrt(sigma)`, giving tilts far above
    `phi` (e.g. a median of 11 degrees for `phi=5`).
    """
    torch.manual_seed(0)
    phi = 5
    transform = RandomTiltAndRotate(phi=phi, theta=180)
    tilts = []
    for i in range(300):
        nag = _nag(seed=i)
        pos = nag[0].pos.clone()
        out = transform(nag)
        tilts.append(_rotation_axis_tilt(pos, out[0].pos))
    tilts = torch.stack(tilts)

    # A 2D gaussian offset falls within 3 standard deviations with
    # probability 1 - exp(-9/2) ~ 0.989
    assert (tilts <= phi).float().mean() > 0.95
