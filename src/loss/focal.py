from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

__all__ = ['WeightedFocalLoss', 'BinaryFocalLoss']


class WeightedFocalLoss(nn.NLLLoss):
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
            weight (Tensor): Weights for each class. Defaults to None.
            gamma (float): A constant, as described in the paper.
                Defaults to 0.
            reduction (str): 'mean' or 'none'.
                Defaults to 'mean'.
            ignore_index (int): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "none".')

        super().__init__(reduction=reduction, ignore_index=ignore_index,weight=weight)

        self.gamma = gamma

    def __repr__(self):
        arg_keys = ['weight', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor, w: Tensor=None) -> Tensor:
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
            # Create copies to avoid in-place operations that can break gradients
            x_neg_mask = x < 0
            x_pos_mask = x > 0
            x_binary[x_neg_mask, 0] = -x[x_neg_mask]
            x_binary[x_pos_mask, 1] = x[x_pos_mask]
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
        y_original_shape = y.shape
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0., device=x.device)
        x = x[unignored_mask]
        w = w[unignored_mask]

        # compute weighted cross entropy term: -weight * log(pt)
        # (weight is already part of super().NLLLoss)
        log_p = F.log_softmax(x, dim=-1)
        ce = super().forward(log_p, y)

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
            # Create a tensor of zeros with original shape and fill valid positions
            loss_with_original_shape = torch.zeros(y_original_shape, dtype=torch.float, device=x.device)
            loss_with_original_shape[unignored_mask] = loss
            return loss_with_original_shape

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
        weight (Sequence): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float): A constant, as described in the paper.
            Defaults to 0.
        reduction (str): 'mean' or 'none'.
            Defaults to 'mean'.
        ignore_index (int): class label to ignore.
            Defaults to -100.
        device (str): Device to move weight to. Defaults to 'cpu'.
        dtype (torch.dtype): dtype to cast weight to.
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


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    
    This class is a simplified version of the WeightedFocalLoss class.
    It supports only 2 classes.
    
    Mean reduction is used.
    """
    
    def __init__(self, 
                 gamma: float = 0, 
                 weight: float = 0.5, 
                 epsilon: float = 1e-6):
        
        super().__init__()
        
        self.gamma = gamma
        self.weight = weight
        self.epsilon = epsilon
        
    def forward(self, p: Tensor, y: Tensor) -> Tensor:
        """
        :param p: (N) Tensor
            Predicted probabilities (for True label)
        :param y: (N) Tensor boolean
            Target labels (True or False)
            
        :return: Tensor
            Loss
        """
        
        
        factor = 2*y.float() - 1 #True -> 1, False -> -1
        p = (~y).float() + p * factor
        p = self.epsilon + (1 - 2 * self.epsilon) * p
        
        
        weight = y.float()*self.weight + (1-y.float()) *(1-self.weight)
        
        loss = self.focal(p) * weight
        
        loss = loss.mean()
        
        return loss
        
    def focal(self, p: Tensor) -> Tensor:
        return -(1 - p) ** self.gamma * torch.log(p)
    
    
    
    
    
    