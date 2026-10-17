import torch
from omegaconf import OmegaConf

from src.data import Data, NAG, Cluster
from src.datamodules.base import BaseDataModule


class _Dataset:
    def __init__(self, root, transform=None, on_device_transform=None, **kwargs):
        self.transform = transform
        self.on_device_transform = on_device_transform


class _DataModule(BaseDataModule):
    _DATASET_CLASS = _Dataset


class _Trainer:
    training = True


def _cloud(num_segments=8, num_points=10):
    """Segments 10 m apart along x, of `num_points` points each."""
    super_index = torch.arange(num_segments).repeat_interleave(num_points)
    pos = torch.rand(num_segments * num_points, 3)
    pos[:, 0] += 10 * super_index
    segments = Data(
        pos=torch.stack([pos[super_index == i].mean(0)
                         for i in range(num_segments)]),
        sub=Cluster(super_index, torch.arange(len(pos)), dense=True))
    return NAG([Data(pos=pos, super_index=super_index), segments])


def test_cpu_sampling_keeps_one_batch_index_per_subgraph():
    """With `cpu_sampling=True`, each cloud of a batch of 2 gives 2
    subgraphs, which must be 4 distinct items of the NAGBatch at every
    level, as with `disjoint=True` on device (GraphNorm and
    QuantizePointCoordinates rely on the 'batch' index)."""
    dm = _DataModule(
        on_device_train_transform=OmegaConf.create([
            {'transform': 'NAGCast'},
            {'transform': 'NodeSize', 'params': {'low': 0}},
            {'transform': 'SampleSubNodes',
             'params': {'low': 0, 'high': 1, 'n_min': 4, 'n_max': 8}},
            {'transform': 'SampleRadiusSubgraphs',
             'params': {'r': 15, 'k': 4, 'k_max': 1000, 'i_level': 1,
                        'disjoint': True, 'cylindrical': True}},
            {'transform': 'SampleSegments', 'params': {'ratio': 0.2}},
            {'transform': 'NAGRestrictSize',
             'params': {'level': '1+', 'num_nodes': 1000}}]),
        dataloader=OmegaConf.create({'batch_size': 2}),
        non_blocking=False,
        cpu_sampling=True)
    dm.setup()
    dm.trainer = _Trainer()

    # What the DataLoader workers return for a batch of 2 clouds
    items = [dm.train_dataset.transform(_cloud()) for _ in range(2)]
    assert [len(x) for x in items] == [2, 2]

    batch = dm.transfer_batch_to_device(items, torch.device('cpu'), 0)
    nag = dm.on_after_batch_transfer(batch, 0)

    for i in range(nag.num_levels):
        assert nag[i].batch.unique().tolist() == [0, 1, 2, 3]
    assert torch.equal(nag[1].batch[nag[0].super_index], nag[0].batch)
