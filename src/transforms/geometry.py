import torch
from src.data import NAG
from src.transforms import Transform
from src.utils.geometry import rodrigues_rotation_matrix


__all__ = [
    'CenterPosition', 'RandomTiltAndRotate', 'RandomAnisotropicScale',
    'RandomAxisFlip']


class CenterPosition(Transform):
    """Center the position of all nodes of all levels of a NAG around
    their level-0 centroid.
    """
    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        offset = nag[0].pos.mean(dim=0)
        for i_level in range(nag.num_levels):
            nag[i_level].pos -= offset
        return nag


class RandomTiltAndRotate(Transform):
    """Rotate the NAG around a random axis, with a random angle. The
    axis is picked following a gaussian jitter around the z axis. The
    angle is picked following a uniform distribution within a specified
    range.

    If the nodes have a `normal` or 'mean_normal' attribute, we also
    rotate those accordingly.

    Warning: any other absolute orientation-related attributes beside
    `pos`, `normal` and 'mean_normal' may be broken by this transform.

    :param phi: float (degrees)
        The random axis will have random angle wrt the z axis. This
        random angle corresponds to adding some random xy offset to z.
         This offset is sampled from a 2D gaussian distribution of
         standard deviation `sigma` computed so that a `3 * sigma` xy
         offset corresponds to a `phi` angle.
    :param theta: float (degrees)
        The random rotation angle will be uniformly picked within
        [-abs(theta), abs(theta)]
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, phi=5, theta=180):
        assert isinstance(phi, (int, float))
        assert isinstance(theta, (int, float))
        self.phi = float(abs(phi))
        self.theta = float(abs(theta))

    def _process(self, nag):
        device = nag.device

        # Generate the random rotation axis
        sigma = self.phi / 180. * torch.pi / 3
        if sigma > 0:
            means = torch.zeros(2, device=device)
            stds = torch.eye(2, device=device) * sigma
            distribution = torch.distributions.MultivariateNormal(means, stds)
            axis_xy = distribution.sample()
            axis_z = torch.ones(1, device=device)
            axis = torch.cat((axis_xy, axis_z))
            axis /= axis.norm()
        else:
            axis = torch.zeros(3, device=device, dtype=torch.float)
            axis[2] = 1

        # Generate the random rotation angle
        theta = torch.rand(1, device=device) * 2 * self.theta - self.theta

        # Compute the rotation matrix
        R = rodrigues_rotation_matrix(axis, theta)

        # Rotate the nodes at each level. If the nodes have a `normal`
        # attribute, we also rotate those accordingly
        for i_level in range(nag.num_levels):
            if sigma <= 0:
                continue
            nag[i_level].pos = nag[i_level].pos @ R.T

            # If the nodes have a `normal` or 'mean_normal' attribute,
            # we also adapt their orientations accordingly
            for k in ['normal', 'mean_normal']:
                if getattr(nag[i_level], k, None) is not None:
                    nag[i_level][k] = self._rotate_normal(nag[i_level][k], R)

            if nag[i_level].edge_attr is not None:
                edge_attr = nag[i_level].edge_attr
                assert edge_attr.shape[1] == 7, \
                    "Expected exactly 7 features in `edge_attr`, generated " \
                    "with `_minimalistic_horizontal_edge_features`"
                dtype = edge_attr.dtype
                edge_attr[:, :3] = (edge_attr[:, :3].float() @ R.T).to(dtype)  # `mean_off`, float16 mm not supported on CPU
                nag[i_level].edge_attr = edge_attr

        return nag

    @staticmethod
    def _rotate_normal(normal, R):
        dtype = normal.dtype
        normal = (normal.float() @ R.T).to(dtype)
        normal[normal[:, 2] < 0] *= -1
        return normal


class RandomAnisotropicScale(Transform):
    """Scales node positions by a randomly sampled factor ``s1, s2, s3``
    within a given interval, *e.g.*, resulting in the following
    transformation matrix

    .. math::
        \left[
        \begin{array}{ccc}
            s1 & 0 & 0 \\
            0 & s2 & 0 \\
            0 & 0 & s3 \\
        \end{array}
        \right]

    for three-dimensional positions.

    If the nodes have a `normal` attribute, we also reorient those
    accordingly, while preserving their unit-norm.

    Warning: any other absolute orientation-related attributes beside
    `pos` and `normal` may be broken by this transform.

    Credit: https://github.com/torch-points3d/torch-points3d

    :param delta: float or List(float)
        Scaling will be uniformly sampled in [-delta, delta]. If a
        3-element list may be passed to scale X, Y and Z differently.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, delta=0.2):
        assert isinstance(delta, (float, int)) or isinstance(delta, (tuple, list))
        if isinstance(delta, (float, int)):
            delta = [float(delta)] * 3
        assert len(delta) == 3
        self.delta = torch.tensor(delta).abs().view(1, -1)

    def _process(self, nag):
        # Generate the random scales
        scale = 1 + (torch.rand(1) * 2 * self.delta - self.delta).to(nag.device)

        for i_level in range(nag.num_levels):
            nag[i_level].pos = nag[i_level].pos * scale

            # If the nodes have a `normal` or 'mean_normal' attribute,
            # we also adapt their orientations accordingly
            for k in ['normal', 'mean_normal']:
                if getattr(nag[i_level], k, None) is not None:
                    nag[i_level][k] = self._scale_normal(nag[i_level][k], scale)

            if getattr(nag[i_level], 'edge_attr', None) is not None:
                edge_attr = nag[i_level].edge_attr
                assert edge_attr.shape[1] == 7, \
                    "Expected exactly 7 features in `edge_attr`, generated " \
                    "with `_minimalistic_horizontal_edge_features`"
                edge_attr[:, :3] *= scale
                edge_attr[:, 3:] *= scale.norm()  # std_off and mean_dist are scaled by the scaling norm, slightly incorrect for std_off...
                nag[i_level].edge_attr = edge_attr

        return nag

    @staticmethod
    def _scale_normal(normal, scale):
        return torch.nn.functional.normalize(normal * scale, dim=1)


class RandomAxisFlip(Transform):
    """Flip the node positions wrt one of the XYZ axes, with a specified
    probability. This transform is not very modular because it is
    intended to be composed with `RandomTiltAndRotate` for richer
    geometric augmentations.

    If the nodes have a `normal` or 'mean_normal' attribute, we also
    flip those accordingly.

    Warning: any other absolute orientation-related attributes beside
    `pos`, `normal` 'mean_normal' may be broken by this transform.

    :param p: float
        Probability of flip
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, axis=0, p=0.5):
        assert isinstance(axis, int)
        assert isinstance(p, float)
        self.axis = axis
        self.p = p

    def _process(self, nag):
        if torch.rand(1, device=nag.device) > self.p:
            return nag

        axis = self.axis
        for i_level in range(nag.num_levels):
            nag[i_level].pos[:, axis] *= -1

            # If the nodes have a `normal` or 'mean_normal' attribute,
            # we also adapt their orientations accordingly
            for k in ['normal', 'mean_normal']:
                if getattr(nag[i_level], k, None) is not None:
                    nag[i_level][k] = self._flip_normal(nag[i_level][k], axis)

            if nag[i_level].edge_attr is not None:
                edge_attr = nag[i_level].edge_attr
                assert edge_attr.shape[1] == 7, \
                    "Expected exactly 7 features in `edge_attr`, generated " \
                    "with `_minimalistic_horizontal_edge_features`"
                edge_attr[:, :3][:, axis] *= -1  # mean_off
                nag[i_level].edge_attr = edge_attr

        return nag

    @staticmethod
    def _flip_normal(normal, axis):
        normal[:, axis] *= -1
        normal[normal[:, 2] < 0] *= -1
        return normal
