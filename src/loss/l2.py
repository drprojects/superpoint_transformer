from torch.nn import MSELoss as TorchL2Loss
from src.loss.weighted import WeightedLossMixIn


__all__ = ['WeightedL2Loss', 'L2Loss']


class WeightedL2Loss(WeightedLossMixIn, TorchL2Loss):
    """Weighted mean squared error (ie L2 loss) between predicted and
    target offsets. This is basically the MSELoss except that positive
    weights must be passed at forward time to give more importance to
    some items.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, reduction='none', **kwargs)


class L2Loss(WeightedL2Loss):
    """Mean squared error (ie L2 loss) between predicted and target
    offsets.

    The forward signature allows using this loss as a weighted loss,
    with input weights ignored.
    """

    def forward(self, input, target, weight):
        return super().forward(input, target, None)
