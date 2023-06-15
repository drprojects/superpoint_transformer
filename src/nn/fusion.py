import torch
from torch import nn


__all__ = ['CatFusion', 'AdditiveFusion', 'TakeFirstFusion', 'TakeSecondFusion']


def fusion_factory(mode):
    """Return the fusion class from an input string.

    :param mode: str
    """
    if mode in ['cat', 'concatenate', 'concatenation', '|']:
        return CatFusion()
    elif mode in ['residual', 'additive', '+']:
        return AdditiveFusion()
    elif mode in ['first', '1', '1st']:
        return TakeFirstFusion()
    elif mode in ['second', '2', '2nd']:
        return TakeSecondFusion()
    else:
        raise NotImplementedError(f"Unknown mode='{mode}'")


class BaseFusion(nn.Module):
    def forward(self, x1, x2):
        if x1 is None and x2 is None:
            return None
        if x1 is None:
            return x2
        if x2 is None:
            return x1
        return self._func(x1, x2)

    def _func(self, x1, x2):
        raise NotImplementedError


class CatFusion(BaseFusion):
    def _func(self, x1, x2):
        return torch.cat((x1, x2), dim=1)


class AdditiveFusion(BaseFusion):
    def _func(self, x1, x2):
        return x1 + x2


class TakeFirstFusion(BaseFusion):
    def _func(self, x1, x2):
        return x1


class TakeSecondFusion(BaseFusion):
    def _func(self, x1, x2):
        return x2

