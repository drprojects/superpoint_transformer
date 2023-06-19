import os
import sys
import torch
import shutil
import logging
from plyfile import PlyData
from src.datasets import BaseDataset
from src.data import Data
from src.datasets.dales_config import *
from torch_geometric.data import extract_tar


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues with DALES on some machines. Hack to
# solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['DALES', 'MiniDALES']


########################################################################
#                                 Utils                                #
########################################################################

def read_dales_tile(
        filepath, xyz=True, intensity=True, semantic=True, instance=False,
        remap=False):
    data = Data()
    key = 'testing'
    with open(filepath, "rb") as f:
        tile = PlyData.read(f)

        if xyz:
            data.pos = torch.stack([
                torch.FloatTensor(tile[key][axis])
                for axis in ["x", "y", "z"]], dim=-1)

        if intensity:
            # Heuristic to bring the intensity distribution in [0, 1]
            data.intensity = torch.FloatTensor(
                tile[key]['intensity']).clip(min=0, max=60000) / 60000

        if semantic:
            y = torch.LongTensor(tile[key]['sem_class'])
            data.y = torch.from_numpy(ID2TRAINID)[y] if remap else y

        if instance:
            data.instance = torch.LongTensor(tile[key]['ins_class'])

    return data


########################################################################
#                                DALES                                 #
########################################################################

class DALES(BaseDataset):
    """DALES dataset.

    Dataset website: https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _form_url = FORM_URL
    _zip_name = OBJECTS_TAR_NAME
    _las_name = LAS_TAR_NAME
    _ply_name = PLY_TAR_NAME
    _unzip_name = OBJECTS_UNTAR_NAME

    @property
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        return DALES_NUM_CLASSES

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return TILES

    def download_dataset(self):
        """Download the DALES Objects dataset.
        """
        # Manually download the dataset
        if not osp.exists(osp.join(self.root, self._zip_name)):
            log.error(
                f"\nDALES does not support automatic download.\n"
                f"Please, register yourself by filling up the form at "
                f"{self._form_url}\n"
                f"From there, manually download the '{self._zip_name}' into "
                f"your '{self.root}/' directory and re-run.\n"
                f"The dataset will automatically be unzipped into the "
                f"following structure:\n"
                f"{self.raw_file_structure}\n"
                f"⛔ Make sure you DO NOT download the "
                f"'{self._las_name}' nor '{self._ply_name}' versions, which "
                f"do not contain all required point attributes.\n")
            sys.exit(1)

        # Unzip the file and rename it into the `root/raw/` directory
        extract_tar(osp.join(self.root, self._zip_name), self.root)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self._unzip_name), self.raw_dir)

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return read_dales_tile(
            raw_cloud_path, intensity=True, semantic=True, instance=False,
            remap=True)

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── raw/
            └── {{train, test}}/
                └── {{tile_name}}.ply
            """

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        if id in self.all_cloud_ids['train']:
            stage = 'train'
        elif id in self.all_cloud_ids['val']:
            stage = 'train'
        elif id in self.all_cloud_ids['test']:
            stage = 'test'
        else:
            raise ValueError(f"Unknown tile id '{id}'")
        return osp.join(stage, self.id_to_base_id(id) + '.ply')

    def processed_to_raw_path(self, processed_path):
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split('/')[-3:]

        # Raw 'val' and 'trainval' tiles are all located in the
        # 'raw/train/' directory
        stage = 'train' if stage in ['trainval', 'val'] else stage

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_path = osp.join(self.raw_dir, stage, base_cloud_id + '.ply')

        return raw_path


########################################################################
#                              MiniDALES                               #
########################################################################

class MiniDALES(DALES):
    """A mini version of DALES with only a few windows for
    experimentation.
    """
    _NUM_MINI = 2

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self):
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
