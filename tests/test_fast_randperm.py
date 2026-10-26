import pytest
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from src.utils import fast_randperm, sparse_sample


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


class _Perms(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return fast_randperm(100)


def _worker_perms():
    pl.seed_everything(0, workers=True)
    loader = DataLoader(_Perms(), batch_size=1, num_workers=2)
    return [batch[0] for batch in loader]


@pytest.mark.parametrize('device', DEVICES)
def test_fast_randperm_is_a_permutation(device):
    perm = fast_randperm(1000, device=device)
    assert perm.device.type == device
    assert perm.dtype == torch.long
    assert torch.equal(perm.sort().values.cpu(), torch.arange(1000))


@pytest.mark.parametrize('device', DEVICES)
def test_fast_randperm_follows_the_seed(device):
    pl.seed_everything(0)
    a = fast_randperm(1000, device=device)
    pl.seed_everything(0)
    b = fast_randperm(1000, device=device)
    assert torch.equal(a, b)


def test_fast_randperm_follows_the_seed_in_workers():
    """Same seed -> same permutations in the DataLoader workers, and the
    workers do not draw the same permutations."""
    a = _worker_perms()
    b = _worker_perms()
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], a[1])


def test_sparse_sample_follows_the_seed_on_cpu():
    idx = torch.randint(0, 50, (10000,))
    pl.seed_everything(0)
    a = sparse_sample(idx, n_max=8)
    pl.seed_everything(0)
    b = sparse_sample(idx, n_max=8)
    assert torch.equal(a, b)
