from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import IterableDataset


__all__ = ['DataLoader']

def __identity__(batch_list):
    """
        fix for windows, where lambda can't be pickled.
        We have to use a top level function
        see:
        https://discuss.pytorch.org/t/cant-pickle-local-object-dataloader-init-locals-lambda/31857/10?page=2
        https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
    """
    return batch_list


class _RepeatSampler:
    """Iterate over `sampler` forever, one pass after the other. A new
    `iter(sampler)` is created for each pass, so a shuffling sampler
    produces a new order at each pass.

    If the sampler has a `set_epoch()` method (eg `DistributedSampler`),
    its epoch is incremented before each new pass. This is needed
    because the next pass starts before Lightning calls `set_epoch()`
    for the next epoch, which would otherwise repeat the same order.

    `len()` is the length of one pass.
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        base = getattr(self.sampler, 'sampler', self.sampler)
        first_pass = True
        while True:
            if not first_pass and callable(getattr(base, 'set_epoch', None)) \
                    and hasattr(base, 'epoch'):
                base.set_epoch(base.epoch + 1)
            first_pass = False
            yield from self.sampler


class DataLoader(TorchDataLoader):
    """Same as torch DataLoader except that the default behavior for
    `collate_fn=None` is a simple identity. (i.e. the DataLoader will
    return a list of elements by default). This approach is meant to
    move the CPU-hungry NAG.from_nag_list (in particular, the level-0
    Data.from_nag_list) to GPU. This is instead taken care of in the
    'DataModule.on_after_batch_transfer' hook, which calls the dataset
    'on_device_transform'.

    Use `collate_fn=NAG.from_data_list` if you want the CPU to do this
    operation (but beware of collisions with our
    'DataModule.on_after_batch_transfer' implementation.

    :param cross_epoch_prefetch: bool
        If True, the workers keep prefetching batches across epoch
        boundaries. By default, PyTorch workers only start loading
        the batches of an epoch when `iter(loader)` is called, even
        with `persistent_workers=True`, so each epoch starts by
        waiting for the first batches. Here, the sampler is repeated
        forever and a single worker iterator is kept alive: each
        `iter(loader)` yields the next `len(loader)` batches from it.
        The shuffling still changes at each epoch, but the order is
        drawn from the main process RNG at a different moment (when
        the workers reach the end of the previous epoch), so a given
        seed does not produce the same order as with
        `cross_epoch_prefetch=False`. If an epoch is not consumed
        entirely (eg `limit_train_batches`, sanity check), the next
        `iter(loader)` restarts a new pass, as without this option.
        Ignored if `num_workers=0` or for an `IterableDataset`
    """
    def __init__(
            self,
            *args,
            collate_fn=None,
            cross_epoch_prefetch=False,
            **kwargs):
        if collate_fn is None:
            collate_fn = __identity__
        super().__init__(*args, collate_fn=collate_fn, **kwargs)
        self.cross_epoch_prefetch = cross_epoch_prefetch
        self._repeat_sampler = None
        self._cross_epoch_iterator = None
        self._cross_epoch_pass_done = False
        if self._use_cross_epoch_prefetch:
            self._repeat_sampler = _RepeatSampler(super()._index_sampler)

    @property
    def _use_cross_epoch_prefetch(self):
        return self.cross_epoch_prefetch \
            and self.num_workers > 0 \
            and not isinstance(self.dataset, IterableDataset)

    @property
    def _index_sampler(self):
        # The worker iterator reads its indices from here
        if self._repeat_sampler is not None:
            return self._repeat_sampler
        return super()._index_sampler

    def __iter__(self):
        if self._repeat_sampler is None:
            return super().__iter__()
        return self._iter_cross_epoch()

    def _iter_cross_epoch(self):
        # Start a new pass if this is the first epoch, or if the
        # previous epoch was not consumed entirely: the worker iterator
        # is then somewhere in the middle of a pass
        if self._cross_epoch_iterator is None \
                or not self._cross_epoch_pass_done:
            self._cross_epoch_iterator = super().__iter__()
        self._cross_epoch_pass_done = False
        for _ in range(len(self)):
            yield next(self._cross_epoch_iterator)
        self._cross_epoch_pass_done = True
