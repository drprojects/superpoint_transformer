import os
import os.path as osp

import pytest

from src.datasets.base import BaseDataset


_IDS = {
    'train': [f'train_{i}' for i in range(5)],
    'val': [f'val_{i}' for i in range(3)],
    'test': [f'test_{i}' for i in range(2)]}


class _CountingDataset(BaseDataset):
    """Counts how often the cloud ids are rebuilt."""
    num_id_builds = 0

    @property
    def all_base_cloud_ids(self):
        type(self).num_id_builds += 1
        return _IDS


def _make_files(root, xy_tiling):
    """Create empty raw and processed files, so that the dataset neither
    downloads nor processes anything."""
    root = osp.join(root, _CountingDataset.__name__.lower())
    t = xy_tiling
    for stage, ids in _IDS.items():
        for ci in ids:
            raw = osp.join(root, 'raw', f'{ci}.ply')
            os.makedirs(osp.dirname(raw), exist_ok=True)
            open(raw, 'w').close()
            for x in range(t):
                for y in range(t):
                    p = osp.join(
                        root, 'processed', stage, 'no_pre_transform',
                        f'{ci}__TILE_{x + 1}-{y + 1}_OF_{t}-{t}.h5')
                    os.makedirs(osp.dirname(p), exist_ok=True)
                    open(p, 'w').close()


@pytest.mark.parametrize('stage', ['train', 'trainval', 'test'])
def test_dataset_paths_are_cached(tmp_path, stage):
    """`cloud_ids` and `processed_paths` are built once, and are the
    same as without caching."""
    _make_files(str(tmp_path), xy_tiling=2)
    dataset = _CountingDataset(str(tmp_path), stage=stage, xy_tiling=2)

    stages = ['train', 'val'] if stage == 'trainval' else [stage]
    expected = [
        osp.join(
            dataset.processed_dir, s, 'no_pre_transform',
            f'{ci}__TILE_{x + 1}-{y + 1}_OF_2-2.h5')
        for s in stages
        for ci in _IDS[s]
        for x in range(2)
        for y in range(2)]
    assert sorted(dataset.processed_paths) == sorted(expected)
    assert len(dataset) == len(expected)

    num_id_builds = _CountingDataset.num_id_builds
    for i in range(len(dataset)):
        dataset.processed_paths[i]
        len(dataset)
    assert _CountingDataset.num_id_builds == num_id_builds
