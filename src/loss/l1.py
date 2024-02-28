from torch.nn import L1Loss as TorchL1Loss
from src.loss.weighted import WeightedLossMixIn


__all__ = ['WeightedL1Loss', 'L1Loss']


class WeightedL1Loss(WeightedLossMixIn, TorchL1Loss):
    """Weighted L1 loss between predicted and target offsets. This is
    basically the L1Loss except that positive weights must be passed at
    forward time to give more importance to some items.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction='none', **kwargs)


class L1Loss(WeightedL1Loss):
    """L1 loss between predicted and target offsets.

    The forward signature allows using this loss as a weighted loss,
    with input weights ignored.
    """

    def forward(self, input, target, weight):
        return super().forward(input, target, None)
