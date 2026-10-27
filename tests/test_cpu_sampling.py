import pytest
import torch
from omegaconf import OmegaConf

from src.data import Data, NAG, Cluster
from src.datamodules.base import BaseDataModule
from src.transforms import SampleRadiusSubgraphs


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


@pytest.mark.parametrize('disjoint', [True, False])
def test_sample_subgraphs_as_list(disjoint):
    sampler = SampleRadiusSubgraphs(
        r=15, k=3, i_level=1, disjoint=disjoint, cylindrical=True,
        as_list=True)
    out = sampler(_cloud())
    assert isinstance(out, list)
    assert len(out) == (3 if disjoint else 1)
    assert all(isinstance(x, NAG) for x in out)

    # Identity when not sampling
    sampler.k = 0
    nag = _cloud()
    assert sampler(nag) == [nag]


def test_train_transform_sampling_keeps_one_batch_index_per_subgraph():
    """When `train_transform` samples 2 subgraphs in each cloud of a
    batch of 2, they must be 4 distinct items of the NAGBatch at every
    level, as with `disjoint=True` on device (GraphNorm and
    QuantizePointCoordinates rely on the 'batch' index)."""
    dm = _DataModule(
        train_transform=OmegaConf.create([
            {'transform': 'NAGCast'},
            {'transform': 'NodeSize', 'params': {'low': 0}},
            {'transform': 'SampleRadiusSubgraphs',
             'params': {'r': 15, 'k': 2, 'k_max': 1000, 'i_level': 1,
                        'disjoint': True, 'cylindrical': True,
                        'as_list': True}},
            {'transform': 'SampleSubNodes',
             'params': {'low': 0, 'high': 1, 'n_min': 4, 'n_max': 8}},
            {'transform': 'SampleSegments', 'params': {'ratio': 0.2}}]),
        on_device_train_transform=OmegaConf.create([
            {'transform': 'NAGRestrictSize',
             'params': {'level': '1+', 'num_nodes': 1000}}]),
        dataloader=OmegaConf.create({'batch_size': 2}),
        non_blocking=False)
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


def _transform_names(transforms):
    return [t.transform for t in transforms or []]


@pytest.mark.parametrize('dataset', ['kitti360', 'dales'])
def test_cpu_sampling_configs(dataset):
    """The `*_ezsp_cpu_sampling` experiments run the same train
    transforms as the `*_ezsp` ones, with the subgraph sampling moved
    to `train_transform`."""
    from src.utils.hydra import init_config
    sampling = ['SampleRadiusSubgraphs', 'SampleSubNodes', 'SampleSegments']
    ref = init_config(overrides=[
        f'experiment=semantic/{dataset}_ezsp']).datamodule
    cpu = init_config(overrides=[
        f'experiment=semantic/{dataset}_ezsp_cpu_sampling']).datamodule

    assert _transform_names(cpu.train_transform) == \
        ['NAGCast', 'NodeSize'] + sampling
    assert cpu.train_transform[2].params.as_list
    assert cpu.train_transform[2].params.k * cpu.dataloader.batch_size \
        == ref.sample_graph_k
    assert cpu.val_transform is None and cpu.test_transform is None
    assert list(cpu.on_device_train_transform) == [
        t for t in ref.on_device_train_transform
        if t.transform not in sampling + ['NAGCast', 'NodeSize']]
    ref_cpu = {t.transform: t for t in ref.on_device_train_transform}
    for t in cpu.train_transform:
        if t.transform != 'SampleRadiusSubgraphs':
            assert t == ref_cpu[t.transform]
    assert cpu.on_device_val_transform == ref.on_device_val_transform
