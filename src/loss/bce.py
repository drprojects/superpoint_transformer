import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss as TorchBCEWithLogitsLoss
from src.loss.weighted import WeightedLossMixIn


__all__ = ['WeightedBCEWithLogitsLoss', 'BCEWithLogitsLoss']


class WeightedBCEWithLogitsLoss(WeightedLossMixIn, TorchBCEWithLogitsLoss):
    """Weighted BCE loss between predicted and target offsets. This is
    basically the BCEWithLogitsLoss except that positive weights must be
    passed at forward time to give more importance to some items.

    Besides, we remove the constraint of passing `pos_weight` as a
    Tensor. This simplifies instantiation with hydra.
    """

    def __init__(self, *args, pos_weight=None, **kwargs):
        if pos_weight is not None and not isinstance(pos_weight, Tensor):
            pos_weight = torch.as_tensor(pos_weight)
        super().__init__(
            *args, pos_weight=pos_weight, reduction='none', **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Normal `load_state_dict` behavior, except for the shared
        `pos_weight`.
        """
        # Get the weight from the state_dict
        pos_weight = state_dict.get('pos_weight')
        state_dict.pop('pos_weight')

        # Normal load_state_dict, ignoring pos_weight
        out = super().load_state_dict(state_dict, strict=strict)

        # Set the pos_weight
        self.pos_weight = pos_weight

        return out


class BCEWithLogitsLoss(WeightedBCEWithLogitsLoss):
    """BCE loss between predicted and target offsets.

    The forward signature allows using this loss as a weighted loss,
    with input weights ignored.
    """

    def forward(self, input, target, weight):
        return super().forward(input, target, None)
