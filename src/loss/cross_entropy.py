import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

__all__ = ['stable_log_softmax', 'CrossEntropyLoss']


def stable_log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """`log_softmax` whose gradient does not collapse when one logit is
    much larger than the others. See "A Fix for Softmax and
    Cross-Entropy", Barron et al., arXiv:2610.31642.

    `F.log_softmax` shifts the logits by their max and computes
    `log(1 + e^-m)`, with `m` the margin of the max logit over the
    log-sum-exp of the others. Once `m` exceeds ~16.6 in float32,
    `1 + e^-m` rounds to 1: the probability of the max logit rounds to
    1 and its gradient to 0, while the gradients of the other logits
    stay positive, so the gradients no longer sum to zero. Here, the
    max logit is excluded from the log-sum-exp and `log(1 + e^-m)` is
    computed as `softplus(-m)`, which stays accurate down to underflow.

    :param x: Tensor
        Logits
    :param dim: int
        Dimension along which the softmax is computed
    :return: Tensor of the same shape as `x`, in float32 if `x` is in
        half precision (like `F.log_softmax` under autocast)
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()

    # With a single class, there are no other logits to compare to
    if x.shape[dim] == 1:
        return F.log_softmax(x, dim=dim)

    # Logits shifted by the max logit, with a constant 0 at the max.
    # NB: the constant matters, with `x - x_max` the max logit would
    # receive a gradient of -1 from its own slot and of ~1 through
    # `x_max`, which cancel out in floating point
    x_max, i_max = x.max(dim=dim, keepdim=True)
    shifted = (x - x_max).scatter(dim, i_max, 0.)

    # Margin of the max logit, excluding it from the log-sum-exp
    others = shifted.scatter(dim, i_max, float('-inf'))
    m = -torch.logsumexp(others, dim=dim, keepdim=True)

    return shifted - F.softplus(-m)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """`nn.CrossEntropyLoss` computed with `stable_log_softmax`, whose
    gradient does not collapse for confidently-predicted items. Same
    parameters and outputs, up to float rounding.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.label_smoothing > 0 or target.is_floating_point():
            raise NotImplementedError(
                f"{type(self).__name__} only supports class index targets "
                f"without label smoothing")
        # NB: `nll_loss` needs contiguous log-probabilities for inputs
        # with spatial dimensions
        log_p = stable_log_softmax(input, dim=1 if input.dim() > 1 else 0)
        return F.nll_loss(
            log_p.contiguous(),
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)
