import torch
import logging
import pandas as pd
from src.utils.geometry import rodrigues_rotation_matrix
from src.datasets.s3dis_config import *
from src.datasets.s3dis import read_s3dis_room, S3DIS

DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['S3DISRoom', 'MiniS3DISRoom']


########################################################################
#                              S3DIS Room                              #
########################################################################

class S3DISRoom(S3DIS):
    """S3DIS dataset, for aligned room-wise prediction.

    Dataset website: http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    fold : `int`
        Integer in [1, ..., 6] indicating the Test Area
    align : `bool`
        Whether the rooms should be canonically aligned, as described in
        section 3.2 of the S3DIS paper
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

    def __init__(self, *args, align=True, **kwargs):
        self.align = align
        super().__init__(*args, **kwargs)

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return {
            'train': [
                f'Area_{i}/{r}' for i in range(1, 7) if i != self.fold
                for r in ROOMS[f'Area_{i}']],
            'val': [
                f'Area_{i}/{r}' for i in range(1, 7) if i != self.fold
                for r in ROOMS[f'Area_{i}']],
            'test': [
                f'Area_{self.fold}/{r}' for r in ROOMS[f'Area_{self.fold}']]}

    def read_single_raw_cloud(self, raw_cloud_path, instance=False):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return read_s3dis_room(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=instance,
            xyz_room=True, align=self.align, is_val=True, verbose=False)

    def processed_to_raw_path(self, processed_path):
        """Given a processed cloud path from `self.processed_paths`,
        return the absolute path to the corresponding raw cloud.

        Overwrite this method if your raw data does not follow the
        default structure.
        """
        # Extract useful information from <path>
        stage, hash_dir, area_id, room_id = \
            osp.splitext(processed_path)[0].split('/')[-4:]
        cloud_id = osp.join(area_id, room_id)

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_ext = osp.splitext(self.raw_file_names_3d[0])[1]
        raw_path = osp.join(self.raw_dir, base_cloud_id + raw_ext)

        return raw_path


########################################################################
#                              MiniS3DIS                               #
########################################################################

class MiniS3DISRoom(S3DISRoom):
    """A mini version of S3DIS with only 2 areas per stage for
    experimentation.
    """
    _NUM_MINI = 1

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
