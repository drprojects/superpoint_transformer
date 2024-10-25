import torch


__all__ = ['is_xyz_tensor']


def is_xyz_tensor(xyz):
    if not isinstance(xyz, torch.Tensor):
        return False
    if not xyz.dim() == 2:
        return False
    return xyz.shape[1] == 3
