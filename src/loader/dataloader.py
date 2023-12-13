from torch.utils.data import DataLoader as TorchDataLoader


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

class DataLoader(TorchDataLoader):
    """Same as torch DataLoader except that the default behaviour for
    `collate_fn=None` is a simple identity. (ie the DataLoader will
    return a list of elements by default). This approach is meant to
    move the CPU-hungry NAG.from_nag_list (in particular, the level-0
    Data.from_nag_list) to GPU. This is instead taken care of in the
    'DataModule.on_after_batch_transfer' hook, which calls the dataset
    'on_device_transform'.

    Use `collate_fn=NAG.from_data_list` if you want the CPU to do this
    operation (but beware of collisions with our
    'DataModule.on_after_batch_transfer' implementation.
    """
    def __init__(self, *args, collate_fn=None, **kwargs):
        if collate_fn is None:
            collate_fn = __identity__
        super().__init__(*args, collate_fn=collate_fn, **kwargs)
