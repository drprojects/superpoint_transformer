from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

__all__ = ['WeightedFocalLoss']


class WeightedFocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.

    Credit: https://github.com/AdeelH/pytorch-multi-class-focal-loss

    Note:
        Modified `alpha` to `weight` to respect the loss template
        expected by `loss_with_target_histogram`
    """

    def __init__(
            self,
            weight: Optional[Tensor] = None,
            gamma: float = 0.,
            reduction: str = 'mean',
            ignore_index: int = -100):
        """Constructor.
        Args:
            weight (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "none".')

        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=weight, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['weight', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor, w: Tensor) -> Tensor:
        """
        :param x: (N, C, ...) Tensor
            Logits
        :param y: (N, ...) Tensor
            Target labels
        :param w: (N, ...) Tensor
            Per-item weights, can be None
        """
        # Convert 1D x to multiclass x. This is an artificial step for
        # binary classification (eg affinity loss), where only 1 score
        # is provided. In this case, we assume that x<0 accounts for y=0
        # and x>0 accounts for y=1 (ie prepared for sigmoid). Here, we
        # convert these precitions to 2D for downstream softmax
        if x.dim() == 1:
            x_binary = torch.zeros(x.shape[0], 2, dtype=x.dtype, device=x.device)
            x_binary[x < 0, 0] = -x[x < 0]
            x_binary[x > 0, 1] = x[x > 0]
            x = x_binary

        # Convert y to long. The NLL loss does not support non-integer
        # target labels
        y = y.long()

        # Convert per-item weights to [0, 1] weights
        if w is None:
            w = torch.ones_like(y).float()
        w = w / w.sum()

        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            w = w.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0., device=x.device)
        x = x[unignored_mask]
        w = w[unignored_mask]

        # compute weighted cross entropy term: -weight * log(pt)
        # (weight is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        log_pt = log_p.gather(dim=1, index=y.view(-1, 1)).squeeze()

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -weight * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        # Apply the per-item weighting
        loss = loss * w

        if self.reduction == 'none':
            return loss

        return loss.sum()


def weighted_focal_loss(
        weight: Optional[Sequence] = None,
        gamma: float = 0.,
        reduction: str = 'mean',
        ignore_index: int = -100,
        device='cpu',
        dtype=torch.float32) -> WeightedFocalLoss:
    """Factory function for WeightedFocalLoss.
    Args:
        weight (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move weight to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast weight to.
            Defaults to torch.float32.
    Returns:
        A WeightedFocalLoss object
    """
    if weight is not None:
        if not isinstance(weight, Tensor):
            weight = torch.tensor(weight)
        weight = weight.to(device=device, dtype=dtype)

    fl = WeightedFocalLoss(
        weight=weight,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl
