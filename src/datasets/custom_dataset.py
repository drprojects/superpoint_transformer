import os.path as osp
import torch
import logging
import laspy
from pathlib import Path
from src.datasets import BaseDataset
from src.data import Data
from src.datasets.custom_dataset_config import CLASS_NAMES, NUM_CLASSES, ID2TRAINID, TILES
import torch.multiprocessing

DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")


__all__ = ["CustomDataset", "MiniCustomDataset"]


########################################################################
#                                 Utils                                #
########################################################################


def read_tile(
    filepath,
):
    data = Data()
    with open(filepath, "rb") as f:
        tile = laspy.read(f)

        data.pos = torch.stack(
            [torch.FloatTensor(tile[ax])/3.28084 for ax in ["x", "y", "z"]],
            dim=-1,
        )

        # Heuristic to bring the intensity distribution in [0, 1]
        data.intensity = (
            torch.FloatTensor(tile["intensity"].astype(float)).clip(min=0, max=60000) / 60000
        )
        
        data.elevation = (
            torch.FloatTensor(tile["hag"].astype(float))
        )

        y = torch.LongTensor([ID2TRAINID[_] for _ in tile["classification"]])
        data.y = y  # type: ignore

    return data


class CustomDataset(BaseDataset):
    """
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

    def __init__(
        self,
        root,
        stage="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        on_device_transform=None,
        save_y_to_csr=True,
        save_pos_dtype=torch.float,
        save_fp_dtype=torch.half,
        xy_tiling=None,
        pc_tiling=None,
        val_mixed_in_train=False,
        test_mixed_in_val=False,
        custom_hash=None,
        in_memory=False,
        point_save_keys=None,
        point_no_save_keys=None,
        point_load_keys=None,
        segment_save_keys=None,
        segment_no_save_keys=None,
        segment_load_keys=None,
        **kwargs,
    ):
        super().__init__(
            root,
            stage,
            transform,
            pre_transform,
            pre_filter,
            on_device_transform,
            save_y_to_csr,
            save_pos_dtype,
            save_fp_dtype,
            xy_tiling,
            pc_tiling,
            val_mixed_in_train,
            test_mixed_in_val,
            custom_hash,
            in_memory,
            point_save_keys,
            point_no_save_keys,
            point_load_keys,
            segment_save_keys,
            segment_no_save_keys,
            segment_load_keys,
            **kwargs,
        )

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
        return NUM_CLASSES

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return TILES

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return read_tile(
            raw_cloud_path,
        )

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── raw/
            └── {{train, test, val}}/
                └── {{tile_name}}.las
            """

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        if id in self.all_cloud_ids["train"]:
            stage = "train"
        elif id in self.all_cloud_ids["val"]:
            stage = "train"
        elif id in self.all_cloud_ids["test"]:
            stage = "test"
        elif id in self.all_cloud_ids["predict"]:
            stage = "predict"
        else:
            raise ValueError(f"Unknown tile id '{id}'")
        return osp.join(stage, self.id_to_base_id(id) + ".las")

    def processed_to_raw_path(self, processed_path):
        """Return the raw cloud path corresponding to the input
        processed path.
        """
        # Extract useful information from <path>
        stage, _, cloud_id = osp.splitext(processed_path)[0].split("/")[-3:]

        # Raw 'val' and 'trainval' tiles are all located in the
        # 'raw/train/' directory
        stage = "train" if stage in ["trainval", "val"] else stage

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = cloud_id

        # Read the raw cloud data
        raw_path = osp.join(self.raw_dir, _, base_cloud_id)

        return raw_path

    def download_dataset(self):
        return None

########################################################################
#                              MiniDALES                               #
########################################################################


class MiniCustomDataset(CustomDataset):
    """A mini version of CustomDataset with only a few windows for
    experimentation.
    """
