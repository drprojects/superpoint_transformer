"""Tests for ``SumLoss`` and the Lovász-softmax loss it is meant for.

``SumLoss`` sums criteria computed on the same logits and labels, eg
the cross-entropy and ``LovaszLoss`` on the first stage of the
multi-stage loss, the other stages keeping the cross-entropy alone.
"""
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from src.loss import (
    CrossEntropyLoss, LovaszLoss, MultiLoss, SumLoss, WeightedFocalLoss)
from src.loss.lovasz import lovasz
from src.utils import loss_with_target_histogram

NUM_CLASSES = 5
CLASS_WEIGHT = torch.tensor([0.5, 1.0, 2.0, 1.0, 3.0])


def random_logits_labels(n, seed=0, scale=2):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, NUM_CLASSES, generator=g) * scale
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    y[::7] = NUM_CLASSES  # ignored items
    return x, y


def value_and_grad(fn, x, *args):
    x = x.clone().requires_grad_()
    loss = fn(x, *args)
    loss.backward()
    return loss.detach(), x.grad


def ce_lovasz(lambdas=(1., 1.)):
    loss = SumLoss([CrossEntropyLoss(), LovaszLoss()], list(lambdas))
    loss.ignore_index = NUM_CLASSES
    return loss


@pytest.mark.parametrize('weight', [None, CLASS_WEIGHT])
@pytest.mark.parametrize('lambdas', [(1., 1.), (0.5, 2.)])
def test_sum_of_criteria(weight, lambdas):
    """The loss and its gradient are the weighted sum of the criteria,
    the class weights only applying to the first one."""
    x, y = random_logits_labels(2000)
    loss = ce_lovasz(lambdas)
    loss.weight = weight
    ce = CrossEntropyLoss(weight=weight, ignore_index=NUM_CLASSES)
    lov = LovaszLoss(ignore_index=NUM_CLASSES)
    expected = lambda x, y: lambdas[0] * ce(x, y) + lambdas[1] * lov(x, y)
    l1, g1 = value_and_grad(loss, x, y)
    l2, g2 = value_and_grad(expected, x, y)
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(g1, g2)


@pytest.mark.parametrize('weight', [None, CLASS_WEIGHT])
def test_matches_focal_lovasz_term(weight):
    """Same loss and gradient as the Lovász term of the ablations, added
    to the 'mean' reduction of `WeightedFocalLoss(gamma=0)` and computed
    on the unignored items with `ignore_index=-1`."""
    x, y = random_logits_labels(2000)
    loss = SumLoss(
        [WeightedFocalLoss(gamma=0), LovaszLoss()], [1., 1.])
    loss.ignore_index = NUM_CLASSES
    loss.weight = weight
    focal = WeightedFocalLoss(
        gamma=0, weight=weight, ignore_index=NUM_CLASSES)

    def ablation(x, y):
        mask = y != NUM_CLASSES
        return focal(x, y) + lovasz(
            x[mask], y[mask], class_to_sum='present', reduction='sum',
            ignore_index=-1)

    l1, g1 = value_and_grad(loss, x, y)
    l2, g2 = value_and_grad(ablation, x, y)
    torch.testing.assert_close(l1, l2)
    torch.testing.assert_close(g1, g2)


def test_ignore_index_and_weight_propagation():
    loss = ce_lovasz()
    assert all(c.ignore_index == NUM_CLASSES for c in loss.criteria)
    loss.weight = CLASS_WEIGHT
    assert torch.equal(loss.weight, CLASS_WEIGHT)
    assert torch.equal(loss.criteria[0].weight, CLASS_WEIGHT)
    assert loss.criteria[1].weight is None


def test_reduction_none_raises():
    """The histogram-based losses need a per-item loss, which a sum of
    criteria with different reductions cannot provide."""
    x, y = random_logits_labels(100)
    y_hist = torch.nn.functional.one_hot(y, NUM_CLASSES + 1)
    with pytest.raises(NotImplementedError):
        loss_with_target_histogram(ce_lovasz(), x, y_hist)


def multi_loss():
    """First stage CE + Lovász, the other 2 stages CE alone."""
    return MultiLoss(
        [ce_lovasz(), CrossEntropyLoss(), CrossEntropyLoss()], [1, 25, 100])


@pytest.mark.parametrize('saved', [None, CLASS_WEIGHT])
@pytest.mark.parametrize('current', [None, CLASS_WEIGHT * 2])
def test_state_dict_round_trip(saved, current):
    """Nested in a `MultiLoss`, the class weights are saved once, under
    `weight`, and loaded back in the first criterion of the `SumLoss`."""
    src = multi_loss()
    src.weight = saved
    state_dict = src.state_dict()
    assert list(state_dict.keys()) == ([] if saved is None else ['weight'])
    dst = multi_loss()
    dst.weight = current
    dst.load_state_dict(state_dict)
    expected = saved if saved is not None else current
    sum_loss = dst.criteria[0]
    if expected is None:
        assert sum_loss.weight is None
    else:
        assert torch.equal(sum_loss.criteria[0].weight, expected)
        assert torch.equal(dst.criteria[1].weight, expected)
    assert sum_loss.criteria[1].weight is None


def test_state_dict_alone():
    loss = ce_lovasz()
    assert len(loss.state_dict()) == 0
    loss.weight = CLASS_WEIGHT
    assert list(loss.state_dict().keys()) == ['weight']
    other = ce_lovasz()
    other.load_state_dict(loss.state_dict())
    assert torch.equal(other.weight, CLASS_WEIGHT)


def test_state_dict_in_parent_module():
    """Nested in a module, the weight is saved under `<prefix>weight`."""
    parent = nn.Module()
    parent.criterion = multi_loss()
    assert len(parent.state_dict()) == 0
    parent.criterion.weight = CLASS_WEIGHT
    assert list(parent.state_dict().keys()) == ['criterion.weight']


def test_instantiate_from_config():
    """The config example of the `SumLoss` docstring."""
    cfg = OmegaConf.create("""
    model:
      multi_stage_loss_lambdas: [1, 25, 100]
      criterion:
        _target_: src.loss.MultiLoss
        criteria:
          - _target_: src.loss.SumLoss
            criteria:
              - _target_: src.loss.CrossEntropyLoss
              - _target_: src.loss.LovaszLoss
            lambdas: [1, 1]
          - _target_: src.loss.CrossEntropyLoss
          - _target_: src.loss.CrossEntropyLoss
        lambdas: ${model.multi_stage_loss_lambdas}
        ignore_index: ${num_classes}
    num_classes: 5
    """)
    loss = instantiate(cfg.model.criterion)
    assert loss.criteria[0].criteria[1].ignore_index == 5
    assert loss.criteria[2].ignore_index == 5
    assert isinstance(loss, MultiLoss) and len(loss) == 3
    assert isinstance(loss.criteria[0], SumLoss)
    assert isinstance(loss.criteria[0].criteria[1], LovaszLoss)
    assert list(loss.lambdas) == [1, 25, 100]


def test_lovasz_softmax_collapse():
    """Past a margin of ~16.6 in float32, `softmax` rounds p to 1 and
    the error of a confidently-right item and its gradient to 0. The
    Lovász-softmax loss matches its float64 reference there."""
    x, y = random_logits_labels(200, scale=1)
    y[y == NUM_CLASSES] = 0
    x[0] = 0
    x[0, y[0]] = 30.
    l32, g32 = value_and_grad(lovasz, x, y)
    l64, g64 = value_and_grad(lovasz, x.double(), y)
    torch.testing.assert_close(l32, l64.float())
    torch.testing.assert_close(g32, g64.float(), rtol=1e-4, atol=1e-12)
    assert g32[0, y[0]] != 0


def test_focal_repr():
    """`weight` is a buffer, not an instance attribute."""
    assert 'gamma=2' in repr(WeightedFocalLoss(gamma=2))
