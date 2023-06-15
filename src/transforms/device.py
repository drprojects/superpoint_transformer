import torch
from src.transforms import Transform
from src.data import NAG


__all__ = ['DataTo', 'NAGTo']


class DataTo(Transform):
    """Move Data object to specified device."""

    def __init__(self, device):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device

    def _process(self, data):
        if data.device == self.device:
            return data
        return data.to(self.device)


class NAGTo(Transform):
    """Move Data object to specified device."""

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, device):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device

    def _process(self, nag):
        if nag.device == self.device:
            return nag
        return nag.to(self.device)
