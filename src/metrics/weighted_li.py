import logging
from torch import Tensor
from typing import Tuple
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.utilities.checks import _check_same_shape


log = logging.getLogger(__name__)


__all__ = ['WeightedL2Error', 'WeightedL1Error', 'L2Error', 'L1Error']


def _weighted_Li_error_update(
        pred: Tensor,
        target: Tensor,
        weight: Tensor,
        norm: int
) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute weighted L1
    error.

    Args:
        pred: Predicted tensor
        target: Ground truth tensor
        weight: weight tensor
        norm: `i` for Li norm (`i` >= 0)
    """
    if weight is not None:
        assert weight.dim() == 1
        assert weight.numel() == pred.shape[0]
    assert norm >= 0

    _check_same_shape(pred, target)

    a = pred - target
    sum_dims = tuple(range(1, a.dim()))
    if norm == 0:
        a = a.any(dim=1).float().sum(dim=sum_dims)
    elif norm == 1:
        a = a.abs().sum(dim=sum_dims)
    else:
        a = a.pow(norm).sum(dim=sum_dims)

    sum_error = (weight * a).sum() if weight is not None else a.sum()
    sum_weight = weight.sum() if weight is not None else pred.shape[0]

    return sum_error, sum_weight


class WeightedL2Error(MeanSquaredError):
    """Simply torchmetrics' MeanSquaredError (ie L2 loss) with
    item-weighted mean to give more importance to some items.
    """

    def update(self, pred: Tensor, target: Tensor, weight: Tensor) -> None:
        """Update state with predictions, targets, and weights."""
        sum_squared_error, sum_weight = _weighted_Li_error_update(
            pred, target, weight, 2)

        self.sum_squared_error += sum_squared_error
        self.total = self.total + sum_weight


class WeightedL1Error(MeanAbsoluteError):
    """Simply torchmetrics' MeanAbsoluteError (ie L1 loss) with
    item-weighted mean to give more importance to some items.
    """

    def update(self, pred: Tensor, target: Tensor, weight: Tensor) -> None:
        """Update state with predictions, targets, and weights."""
        sum_abs_error, sum_weight = _weighted_Li_error_update(
            pred, target, weight, 1)

        self.sum_abs_error += sum_abs_error
        self.total = self.total + sum_weight


class L2Error(WeightedL2Error):
    """Simply torchmetrics' MeanSquaredError (ie L2 loss) with summation
    instead of mean along the feature dimensions.
    """
    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        super().update(pred, target, None)


class L1Error(WeightedL1Error):
    """Simply torchmetrics' MeanAbsoluteError (ie L1 loss) with
    summation instead of mean along the feature dimensions.
    """
    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        super().update(pred, target, None)
