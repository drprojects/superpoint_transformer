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
