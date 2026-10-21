import logging
import os
import os.path as osp
import random

import pytest
import torch

from src.data import NAG, Data
from src.datasets.base import BaseDataset
from src.transforms import SampleXYTiling, SampleRecursiveMainXYAxisTiling


RAW_IDS = ['cloud_a', 'cloud_b']


def _raw_cloud(raw_path):
    """Deterministic toy raw cloud for a given raw path."""
    seed = RAW_IDS.index(osp.splitext(osp.basename(raw_path))[0])
    g = torch.Generator().manual_seed(seed)
    return Data(pos=torch.rand(3000, 3, generator=g) * 30)


class _ToyDataset(BaseDataset):
    """Minimal dataset counting how many times each raw cloud is read.
    """

    def __init__(self, *args, **kwargs):
        self.raw_reads = []
        super().__init__(*args, **kwargs)

    @property
    def class_names(self):
        return ['a', 'void']

    @property
    def num_classes(self):
        return 1

    @property
    def all_base_cloud_ids(self):
        return {'train': RAW_IDS, 'val': [], 'test': []}

    def download_dataset(self):
        pass

    def read_single_raw_cloud(self, raw_cloud_path):
        self.raw_reads.append(raw_cloud_path)
        return _raw_cloud(raw_cloud_path)


class _ShuffledToyDataset(_ToyDataset):
    """Processes the tiles in a shuffled order, so the tiles of a raw
    cloud are not consecutive and the raw cloud cache is useless.
    """

    @property
    def processed_file_names(self):
        names = list(super().processed_file_names)
        random.Random(0).shuffle(names)
        return names


def _make_root(tmp_path, cls=_ToyDataset):
    root = str(tmp_path)
    raw_dir = osp.join(root, cls.__name__.lower(), 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    for i in RAW_IDS:
        open(osp.join(raw_dir, f'{i}.ply'), 'w').close()
    return root


# Tiling kwargs and the number of tiles they make per raw cloud
TILINGS = [({'xy_tiling': 3}, 9), ({'pc_tiling': 2}, 4)]


def _tile(dataset, path, raw):
    """Tile `raw` the way `_process_single_cloud` does for `path`."""
    tile = dataset.get_tile_from_path(path)[0]
    if dataset.xy_tiling is not None:
        return SampleXYTiling(x=tile[0], y=tile[1], tiling=tile[2])(raw)
    return SampleRecursiveMainXYAxisTiling(x=tile[0], steps=tile[1])(raw)


@pytest.mark.parametrize('tiling, num_tiles', TILINGS)
def test_tiling_reads_each_raw_cloud_once(tmp_path, caplog, tiling, num_tiles):
    """With `xy_tiling` or `pc_tiling`, the tiles of a raw cloud must
    read it only once, and the saved tiles must be the same as without
    the cache.
    """
    with caplog.at_level(logging.INFO):
        dataset = _ToyDataset(_make_root(tmp_path), **tiling)

    num_saved = 2 * (num_tiles - 1)
    assert len(dataset) == 2 * num_tiles
    assert len(dataset.raw_reads) == 2
    assert f'with 2 raw cloud reads ({num_saved} reads saved)' in caplog.text
    assert 'not processed consecutively' not in caplog.text
    assert dataset._raw_cloud_cache is None

    for path in dataset.processed_paths:
        raw = _raw_cloud(dataset.processed_to_raw_path(path))
        expected = _tile(dataset, path, raw)
        saved = NAG.load(path)[0]
        assert torch.equal(saved.pos, expected.pos)


@pytest.mark.parametrize('tiling, num_tiles', TILINGS)
def test_tiling_warns_when_tiles_not_consecutive(
        tmp_path, caplog, tiling, num_tiles):
    """If the tiles of a raw cloud are not processed one after the
    other, the cache saves (almost) nothing, and a warning says so.
    """
    with caplog.at_level(logging.INFO):
        dataset = _ShuffledToyDataset(
            _make_root(tmp_path, _ShuffledToyDataset), **tiling)

    assert len(dataset.raw_reads) > 2
    assert 'not processed consecutively' in caplog.text


def test_no_raw_cloud_cache_without_tiling(tmp_path, caplog):
    """The raw cloud cache is only used with tiling: without it, each
    processed cloud reads its raw cloud.
    """
    with caplog.at_level(logging.INFO):
        dataset = _ToyDataset(_make_root(tmp_path))

    assert len(dataset.raw_reads) == len(dataset) == 2
    assert 'raw cloud cache' not in caplog.text
