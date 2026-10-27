"""Tests for ``stable_log_softmax`` and ``CrossEntropyLoss``.

In float32, ``F.log_softmax`` rounds the probability of a logit to 1
once its margin over the other logits exceeds ~16.6, which zeroes its
gradient while the other gradients stay positive ("Softmax Collapse",
Barron et al., arXiv:2610.31642). The fixed functions must match
``F.cross_entropy`` in the usual regime, and the float64 reference
beyond it.
"""
import pytest
import torch
import torch.nn.functional as F

from src.loss import CrossEntropyLoss, WeightedFocalLoss, stable_log_softmax

NUM_CLASSES = 5
CLASS_WEIGHT = torch.tensor([0.5, 1.0, 2.0, 1.0, 3.0])


def random_logits_labels(n, seed=0, scale=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, NUM_CLASSES, generator=g) * scale
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return x, y


def value_and_grad(fn, x, *args):
    x = x.clone().requires_grad_()
    loss = fn(x, *args)
    loss.sum().backward()
    return loss.detach(), x.grad


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
@pytest.mark.parametrize('weight', [None, CLASS_WEIGHT])
def test_matches_cross_entropy(reduction, weight):
    x, y = random_logits_labels(2000)
    y[::7] = NUM_CLASSES  # ignored items
    kwargs = dict(weight=weight, ignore_index=NUM_CLASSES, reduction=reduction)
    loss, grad = value_and_grad(CrossEntropyLoss(**kwargs), x, y)
    loss_ref, grad_ref = value_and_grad(
        lambda a, b: F.cross_entropy(a, b, **kwargs), x, y)
    torch.testing.assert_close(loss, loss_ref)
    torch.testing.assert_close(grad, grad_ref)


def test_matches_cross_entropy_spatial():
    x, y = random_logits_labels(24)
    x = x.view(2, 3, 4, NUM_CLASSES).permute(0, 3, 1, 2)
    y = y.view(2, 3, 4)
    loss, grad = value_and_grad(CrossEntropyLoss(), x, y)
    loss_ref, grad_ref = value_and_grad(F.cross_entropy, x, y)
    torch.testing.assert_close(loss, loss_ref)
    torch.testing.assert_close(grad, grad_ref)


def reference_log_softmax(x):
    """float64 log-softmax with `log1p`, accurate beyond the float64
    precision cliff of `F.log_softmax` (margin ~36.7).
    """
    x = x.double()
    shifted = x - x.max(dim=-1, keepdim=True).values
    is_max = shifted == 0
    others = shifted.exp().masked_fill(is_max, 0).sum(dim=-1, keepdim=True)
    return shifted - torch.log1p(others)


@pytest.mark.parametrize('margin', [5., 15., 20., 30., 60.])
def test_log_softmax_matches_float64(margin):
    """Relative precision of the log-probabilities and of their
    gradient, for a max logit far above the others.
    """
    x, _ = random_logits_labels(100, scale=0.5)
    x[:, 0] += margin
    # Random cotangent, to check the full Jacobian. NB: it is seeded, as
    # x, since a gradient component cancelling out to ~1e-3 of its row
    # would exceed `rtol` by float32 rounding alone
    g = torch.Generator().manual_seed(1)
    v = torch.randn(x.shape, generator=g)
    out, grad = value_and_grad(lambda a: stable_log_softmax(a) * v, x)
    _, grad_ref = value_and_grad(
        lambda a: F.log_softmax(a, dim=-1) * v.double(), x.double())
    out_ref = reference_log_softmax(x) * v
    torch.testing.assert_close(out, out_ref.float(), rtol=1e-5, atol=0)
    torch.testing.assert_close(grad, grad_ref.float(), rtol=1e-4, atol=0)


@pytest.mark.parametrize('criterion', [
    CrossEntropyLoss(reduction='none'),
    WeightedFocalLoss(gamma=0, reduction='none'),
    WeightedFocalLoss(gamma=2, reduction='none')],
    ids=['ce', 'focal_gamma0', 'focal_gamma2'])
def test_no_collapse(criterion):
    """Example of the paper: logits [9, -9, -9], target 0. The margin
    m = 18 - log(2) is beyond the float32 precision cliff (~16.6): the
    loss and the target gradient must not round to 0, and the gradients
    must sum to 0.
    """
    x = torch.tensor([[9., -9., -9.]])
    y = torch.tensor([0])
    loss, grad = value_and_grad(criterion, x, y)
    loss_ref, grad_ref = value_and_grad(
        lambda a, b: criterion(a, b), x.double(), y)
    assert loss.item() > 0
    assert grad[0, 0].item() < 0
    torch.testing.assert_close(loss, loss_ref.float(), rtol=1e-5, atol=0)
    torch.testing.assert_close(grad, grad_ref.float(), rtol=1e-4, atol=0)
    assert grad.sum().abs() <= 1e-6 * grad.abs().max()


def test_single_class():
    x = torch.randn(10, 1)
    torch.testing.assert_close(stable_log_softmax(x), F.log_softmax(x, dim=-1))


def test_half_precision_input():
    """Like `F.cross_entropy` under autocast, half-precision logits are
    handled in float32.
    """
    x, y = random_logits_labels(100)
    loss = CrossEntropyLoss()(x.bfloat16(), y)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, F.cross_entropy(x.bfloat16().float(), y))
