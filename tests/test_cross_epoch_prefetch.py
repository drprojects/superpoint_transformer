import time

import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler

from src.loader import DataLoader


class _Range(Dataset):
    def __init__(self, n, delay=0):
        self.n = n
        self.delay = delay

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        time.sleep(self.delay)
        return i


def _epochs(loader, n_epochs):
    return [sorted_flat(loader, ordered=True) for _ in range(n_epochs)]


def sorted_flat(loader, ordered=False):
    items = [i for batch in loader for i in batch]
    return items if ordered else sorted(items)


@pytest.mark.parametrize('persistent_workers', [True, False])
@pytest.mark.parametrize('shuffle', [True, False])
def test_cross_epoch_prefetch_epochs(shuffle, persistent_workers):
    """Each epoch yields `len(loader)` batches covering the dataset once,
    with a new order at each epoch when shuffling."""
    n = 10
    loader = DataLoader(
        _Range(n), batch_size=3, shuffle=shuffle, num_workers=2,
        persistent_workers=persistent_workers, cross_epoch_prefetch=True)
    assert len(loader) == 4

    epochs = _epochs(loader, 4)
    for items in epochs:
        assert len(items) == n
        assert sorted(items) == list(range(n))
    if shuffle:
        assert len({tuple(e) for e in epochs}) > 1
    else:
        assert all(e == list(range(n)) for e in epochs)


def test_cross_epoch_prefetch_partial_epoch():
    """An epoch stopped early (eg Lightning sanity check) must not shift
    the next epochs."""
    n = 10
    loader = DataLoader(
        _Range(n), batch_size=3, num_workers=2, persistent_workers=True,
        cross_epoch_prefetch=True)
    next(iter(loader))
    for _ in range(3):
        assert sorted_flat(loader, ordered=True) == list(range(n))


def test_cross_epoch_prefetch_distributed_sampler():
    """`DistributedSampler` must change its order at each epoch, even
    though the next pass starts before `set_epoch()` is called."""
    n = 32
    ds = _Range(n)
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True)
    loader = DataLoader(
        ds, batch_size=4, sampler=sampler, num_workers=2,
        persistent_workers=True, cross_epoch_prefetch=True)

    epochs = []
    for epoch in range(3):
        # Lightning calls set_epoch at the start of each epoch
        sampler.set_epoch(epoch)
        epochs.append(sorted_flat(loader, ordered=True))
    assert len({tuple(e) for e in epochs}) == 3

    # Same orders as without cross-epoch prefetch
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=True)
    ref = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=0)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        assert sorted_flat(ref, ordered=True) == epochs[epoch]


def test_cross_epoch_prefetch_no_worker():
    """Ignored with `num_workers=0`."""
    loader = DataLoader(
        _Range(5), batch_size=2, num_workers=0, cross_epoch_prefetch=True)
    assert len(loader) == 3
    assert _epochs(loader, 2) == [list(range(5))] * 2


def test_cross_epoch_prefetch_first_batch_ready():
    """The first batch of an epoch is loaded while the previous epoch is
    consumed."""
    delay = 0.2
    loader = DataLoader(
        _Range(4, delay=delay), batch_size=1, num_workers=2,
        persistent_workers=True, cross_epoch_prefetch=True)
    sorted_flat(loader)
    time.sleep(2 * delay)  # time for the workers to prefetch
    t0 = time.perf_counter()
    next(iter(loader))
    assert time.perf_counter() - t0 < delay / 2
