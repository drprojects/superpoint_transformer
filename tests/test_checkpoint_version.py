from types import SimpleNamespace

import pytest
import torch

from src import __version__
from src.models.semantic import SemanticSegmentationModule
from src.transforms import PretrainedCNN
from src.utils import (
    MIN_CHECKPOINT_VERSION,
    VersionHolder,
    check_checkpoint_version)


def test_current_version_is_compatible():
    check_checkpoint_version(__version__)


@pytest.mark.parametrize('version', ['2.1.0', '3.0.0'])
def test_old_version_raises(version):
    with pytest.raises(ValueError, match=MIN_CHECKPOINT_VERSION):
        check_checkpoint_version(version)


def test_old_version_allowed(caplog):
    check_checkpoint_version('3.0.0', allow_old_checkpoint=True)
    assert 'allow_old_checkpoint=True' in caplog.text


def _on_load_checkpoint(metadata, allow_old_checkpoint=False):
    """Call `SemanticSegmentationModule.on_load_checkpoint()` on a
    minimal stand-in for the module."""
    net = SimpleNamespace(
        version=None, version_holder=VersionHolder(__version__))
    module = SimpleNamespace(
        net=net,
        hparams=SimpleNamespace(allow_old_checkpoint=allow_old_checkpoint))
    checkpoint = {'metadata': metadata} if metadata is not None else {}
    SemanticSegmentationModule.on_load_checkpoint(module, checkpoint)
    return module


@pytest.mark.parametrize('metadata', [None, {'__version__': '3.0.0'}])
def test_module_rejects_old_checkpoint(metadata):
    with pytest.raises(ValueError, match=MIN_CHECKPOINT_VERSION):
        _on_load_checkpoint(metadata)


def test_module_loads_old_checkpoint_if_allowed():
    module = _on_load_checkpoint(
        {'__version__': '3.0.0'}, allow_old_checkpoint=True)
    assert module.net.version == '3.0.0'


def test_module_loads_current_checkpoint():
    module = _on_load_checkpoint({'__version__': __version__})
    assert module.net.version == __version__


def test_pretrained_cnn_loads_old_checkpoint(tmp_path):
    """The EZ-SP partition CNN is not affected by the 3.1.0 fixes, so
    its checkpoints are loaded whatever their version."""
    first_stage = torch.nn.Linear(3, 2)
    path = str(tmp_path / 'cnn.ckpt')
    torch.save({
        'metadata': {'__version__': '3.0.0'},
        'state_dict': {
            f'net.first_stage.{k}': v
            for k, v in first_stage.state_dict().items()}}, path)
    loaded = PretrainedCNN.load_checkpoint(torch.nn.Linear(3, 2), path, 'cpu')
    assert torch.equal(loaded.weight, first_stage.weight)
