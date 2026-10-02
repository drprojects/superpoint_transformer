"""Regression tests for ``WeightedFocalLoss`` reductions.

Both call paths used by the semantic module are covered:
- the 'mean' path, ``criterion(logits, labels)`` (level-1 loss in
  ``ce_kl``);
- the histogram path, ``loss_with_target_histogram(criterion, ...)``
  (levels 2-3 in ``ce_kl``, 'kl' / 'wce' losses), which
  switches the criterion to ``reduction='none'`` and applies its own
  normalized per-sample weights.
"""
import pytest
import torch
import torch.nn.functional as F

from src.loss.focal import WeightedFocalLoss
from src.utils.loss import loss_with_target_histogram

NUM_CLASSES = 5


def reference_focal(x, y, gamma, class_weight=None, sample_weight=None):
    """Σ_i a_i·(1-p_i)^γ·(-log p_i) / Σ_i a_i with a_i = s_i·c_{y_i}.

    With gamma=0 this is exactly ``F.cross_entropy``'s weighted mean.
    """
    log_p = F.log_softmax(x, dim=-1)
    log_pt = log_p.gather(1, y[:, None])[:, 0]
    a = torch.ones_like(log_pt)
    if class_weight is not None:
        a = a * class_weight[y]
    if sample_weight is not None:
        a = a * sample_weight.float()
    focal = (1 - log_pt.exp()) ** gamma
    return (a * focal * -log_pt).sum() / a.sum()


def random_logits_labels(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, NUM_CLASSES, generator=g) * 2
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return x, y


def random_histogram(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, NUM_CLASSES, generator=g) * 2
    hist = torch.randint(0, 50, (n, NUM_CLASSES), generator=g)
    hist = hist * (torch.rand(n, NUM_CLASSES, generator=g) < 0.4)
    hist[hist.sum(dim=1) == 0, 0] = 1
    return x, hist


CLASS_WEIGHT = torch.tensor([0.5, 1.0, 2.0, 1.0, 3.0])


@pytest.mark.parametrize('class_weight', [None, CLASS_WEIGHT])
def test_mean_gamma0_matches_cross_entropy(class_weight):
    x, y = random_logits_labels(2000)
    loss = WeightedFocalLoss(gamma=0, weight=class_weight)(x, y)
    expected = F.cross_entropy(x, y, weight=class_weight)
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize('gamma', [1, 2, 3])
@pytest.mark.parametrize('class_weight', [None, CLASS_WEIGHT])
def test_mean_matches_reference(gamma, class_weight):
    x, y = random_logits_labels(2000)
    loss = WeightedFocalLoss(gamma=gamma, weight=class_weight)(x, y)
    expected = reference_focal(x, y, gamma, class_weight)
    torch.testing.assert_close(loss, expected)


def test_mean_ignores_ignore_index():
    x, y = random_logits_labels(2000)
    y[::3] = NUM_CLASSES
    crit = WeightedFocalLoss(gamma=2, ignore_index=NUM_CLASSES)
    keep = y != NUM_CLASSES
    expected = reference_focal(x[keep], y[keep], 2)
    torch.testing.assert_close(crit(x, y), expected)


def test_none_returns_unnormalized_per_item_loss():
    x, y = random_logits_labels(100)
    y[::7] = NUM_CLASSES
    crit = WeightedFocalLoss(
        gamma=2, weight=CLASS_WEIGHT, reduction='none',
        ignore_index=NUM_CLASSES)
    loss = crit(x, y)
    assert loss.shape == y.shape
    keep = y != NUM_CLASSES
    assert torch.all(loss[~keep] == 0)
    log_pt = F.log_softmax(x[keep], -1).gather(1, y[keep][:, None])[:, 0]
    expected = (
        CLASS_WEIGHT[y[keep]] * (1 - log_pt.exp()) ** 2 * -log_pt)
    torch.testing.assert_close(loss[keep], expected)


@pytest.mark.parametrize('n', [100, 2000, 20000])
@pytest.mark.parametrize('class_weight', [None, CLASS_WEIGHT])
def test_histogram_gamma0_matches_cross_entropy(n, class_weight):
    """Focal(γ=0) must be a drop-in replacement for CE on this path."""
    x, hist = random_histogram(n)
    loss = loss_with_target_histogram(
        WeightedFocalLoss(gamma=0, weight=class_weight), x, hist)
    expected = loss_with_target_histogram(
        torch.nn.CrossEntropyLoss(weight=class_weight), x, hist)
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize('n', [100, 2000])
def test_histogram_matches_reference(n):
    x, hist = random_histogram(n)
    crit = WeightedFocalLoss(gamma=2)
    loss = loss_with_target_histogram(crit, x, hist)
    mask = hist != 0
    expected = reference_focal(
        x.repeat_interleave(mask.sum(dim=1), dim=0),
        torch.where(mask)[1],
        2,
        sample_weight=hist[mask])
    torch.testing.assert_close(loss, expected)
    # the helper must restore the criterion's reduction
    assert crit.reduction == 'mean'


def test_focal_downweights_easy_examples_per_sample():
    """Each sample's gradient is modulated by its own (1-p)^γ."""
    # true class 0 for all: easy, medium, hard
    x = torch.tensor([[4., 0.], [0.5, 0.], [-3., 0.]], requires_grad=True)
    y = torch.zeros(3, dtype=torch.long)
    WeightedFocalLoss(gamma=2)(x, y).backward()
    g_focal = x.grad[:, 0].abs()
    x.grad = None
    F.cross_entropy(x, y).backward()
    g_ce = x.grad[:, 0].abs()
    ratio_focal = g_focal[2] / g_focal[0]
    ratio_ce = g_ce[2] / g_ce[0]
    assert ratio_focal > 100 * ratio_ce


def test_single_sample_and_binary_logits():
    x, y = random_logits_labels(1)
    torch.testing.assert_close(
        WeightedFocalLoss(gamma=2)(x, y), reference_focal(x, y, 2))
    # 1D logits: binary path (x>0 -> class 1)
    x1 = torch.tensor([2., -1., 0.5, -3.])
    y1 = torch.tensor([1, 0, 0, 1])
    loss = WeightedFocalLoss(gamma=0)(x1, y1)
    x2 = torch.zeros(4, 2)
    x2[x1 < 0, 0] = -x1[x1 < 0]
    x2[x1 > 0, 1] = x1[x1 > 0]
    torch.testing.assert_close(loss, F.cross_entropy(x2, y1))
