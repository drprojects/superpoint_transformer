import torch
import numpy as np


__all__ = [
    'cross_product_matrix', 'rodrigues_rotation_matrix', 'base_vectors_3d']


def cross_product_matrix(k):
    """Compute the cross-product matrix of a vector k.

    Credit: https://github.com/torch-points3d/torch-points3d
    """
    return torch.tensor(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=k.device)


def rodrigues_rotation_matrix(axis, theta_degrees):
    """Given an axis and a rotation angle, compute the rotation matrix
    using the Rodrigues formula.

    Source : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    Credit: https://github.com/torch-points3d/torch-points3d
    """
    axis = axis / axis.norm()
    K = cross_product_matrix(axis)
    t = torch.tensor([theta_degrees / 180. * np.pi], device=axis.device)
    R = torch.eye(3, device=axis.device) \
        + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def base_vectors_3d(x):
    """Compute orthonormal bases for a set of 3D vectors. The 1st base
    vector is the normalized input vector, while the 2nd and 3rd vectors
    are constructed in the corresponding orthogonal plane. Note that
    this problem is underconstrained and, as such, any rotation of the
    output base around the 1st vector is a valid orthonormal base.
    """
    assert x.dim() == 2
    assert x.shape[1] == 3

    # First direction is along x
    a = x

    # If x is 0 vector (norm=0), arbitrarily put a to (1, 0, 0)
    a[torch.where(a.norm(dim=1) == 0)[0]] = torch.tensor(
        [[1, 0, 0]], dtype=x.dtype, device=x.device)

    # Safely normalize a
    a = a / a.norm(dim=1).view(-1, 1)

    # Build a vector orthogonal to a
    b = torch.vstack((a[:, 1] - a[:, 2], a[:, 2] - a[:, 0], a[:, 0] - a[:, 1])).T

    # In the same fashion as when building a, the second base vector
    # may be 0 by construction (ie a is of type (v, v, v)). So we need
    # to deal with this edge case by setting
    b[torch.where(b.norm(dim=1) == 0)[0]] = torch.tensor(
        [[2, 1, -1]], dtype=x.dtype, device=x.device)

    # Safely normalize b
    b /= b.norm(dim=1).view(-1, 1)

    # Cross product of a and b to build the 3rd base vector
    c = torch.linalg.cross(a, b)

    return torch.cat((a.unsqueeze(1), b.unsqueeze(1), c.unsqueeze(1)), dim=1)
