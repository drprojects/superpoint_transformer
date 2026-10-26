import pytest
import torch
from torch import nn

from src.loss import MultiLoss


def _loss(weight=None):
    loss = MultiLoss([nn.CrossEntropyLoss(), nn.CrossEntropyLoss()], [1., 1.])
    if weight is not None:
        loss.weight = weight
    return loss


def test_state_dict_without_class_weights():
    """Before `on_fit_start` sets the class weights, there is no weight
    to save."""
    assert len(_loss().state_dict()) == 0


@pytest.mark.parametrize('saved', [None, torch.rand(3)])
@pytest.mark.parametrize('current', [None, torch.rand(3)])
def test_state_dict_round_trip(saved, current):
    """A saved weight is loaded, otherwise the current one is kept."""
    loss = _loss(current)
    loss.load_state_dict(_loss(saved).state_dict())
    expected = saved if saved is not None else current
    if expected is None:
        assert loss.weight is None
    else:
        assert all(torch.equal(c.weight, expected) for c in loss.criteria)


def test_load_state_dict_old_format():
    """Weights saved under `criteria.i.weight` are still loaded."""
    weight = torch.rand(3)
    state_dict = {'criteria.0.weight': weight, 'criteria.1.weight': weight}
    loss = _loss(torch.rand(3))
    loss.load_state_dict(state_dict)
    assert all(torch.equal(c.weight, weight) for c in loss.criteria)


def test_state_dict_in_parent_module():
    """Nested in a module, the weight is saved under `<prefix>weight`."""
    weight = torch.rand(3)
    parent = nn.Module()
    parent.criterion = _loss()
    assert len(parent.state_dict()) == 0
    parent.criterion.weight = weight
    state_dict = parent.state_dict()
    assert list(state_dict.keys()) == ['criterion.weight']
    assert torch.equal(state_dict['criterion.weight'], weight)
