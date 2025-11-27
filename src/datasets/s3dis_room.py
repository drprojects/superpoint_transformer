import logging
import os
import os.path as osp
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

    :param root: str
        Root directory where the dataset should be saved.
    :param stage : {'train', 'val', 'test', 'trainval'}
    :param transform: Transform
        Function operating on data. This is executed in the
        dataloader, on CPU. In order to maximize dataloader throughput,
        we tend to postpone all costly operations to rather happen
        on-device with `on_device_transform`. If you still prefer
        running some things on CPU, be careful of what you put in
        `transform`. In case `in_memory=True`, the `transform` will only
        be executed once upon initial data loading to RAM (see
        `in_memory`)
    :param pre_transform: Transform
        Function operating on data. This is called only once at dataset
        preprocessing time to do all the heavy lifting of data
        preparation once and for all
    :param pre_filter: Transform
        Function operating on data. This is called only once at dataset
        preprocessing time, after `pre_transform`, to fiter out some
        data objects before saving
    :param on_device_transform: Transform
        Function operating on data, called in the
        'on_after_batch_transfer' hook. This is where device-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders (see `transform`)
    :param save_y_to_csr: bool
        Whether to save 'y' semantic segmentation label histograms using
        a custom CSR format to save memory and I/O time
    :param save_pos_dtype: torch.dtype
        Torch dtype to which 'pos' should be saved to disk. By default,
        we use float32 precision. This is usually sufficient even for
        dealing with high-resolution point clouds. Importantly, if the
        point coordinates in your raw point clouds are expressed as
        float64, it is possible to maintain high-precision localization
        without manipulating float64 coordinates. To this end, we use a
        small float64 'pos_offset' tensor attribute in your Data objects
        (see how this is done in `read_dales_tile`, for instance)
    :param save_fp_dtype: torch.dtype
        Torch dtype to which all floating point tensors (other than
        'pos' and 'pos_offset') should be saved to disk. Unless the
        associated tensors require very high precision, we recommend
        using float16 to save memory and I/O time
    :param load_non_fp_to_long: bool
        Non-floating-point tensors are saved with the smallest
        precision-preserving dtype possible. `load_non_fp_to_long` rules
        whether these should be cast back to int64 upon reading. To save
        memory and I/O time, we recommend setting
        `load_non_fp_to_long=False` and using the `Cast` or `NAGCast`
        Transform in your `on_device_transform`. This allows postponing
        the casting to GPU and accelerates reading from disk and CPU-GPU
        transfer for the DataLoader
    :param xy_tiling: int
        If provided, the raw point cloud tiles will be split into
        smaller sub-tiles before calling `pre_transform` at
        preprocessing time. This allows chunking very large clouds into
        more manageable pieces, which can alleviate the memory cost of
        some preprocessing and inference operations when CPU/GPU RAM is
        scarce. When using `xy_tiling`, each raw input cloud will be
        split into `xy_tiling * xy_tiling` tiles, based on a regular XY
        grid. Note that this is blind to the orientation and shape of
        your cloud and is typically recommended for densely sampled,
        square cloud tiles (e.g. the DALES dataset). For a tiling that
        better follows the XY structure of a cloud, see `pc_tiling`
    :param pc_tiling: int
        If provided, the raw point cloud tiles will be split into
        smaller sub-tiles before calling `pre_transform` at
        preprocessing time. This allows chunking very large clouds into
        more manageable pieces, which can alleviate the memory cost of
        some preprocessing and inference operations when CPU/GPU RAM is
        scarce. When using `pc_tiling`, each raw input cloud will be
        recursively split into `2^pc_tiling` tiles of point counts,
        based on the principal component of the cloud's XY coordinates
    :param val_mixed_in_train: bool
        Whether the 'val' stage data is saved in the same clouds as the
        'train' stage. This may happen when the stage splits are
        performed inside the clouds. In this case, an
        `on_device_transform` will be automatically created to separate
        stage-specific data upon reading
    :param test_mixed_in_val: bool
        Whether the 'test' stage data is saved in the same clouds as the
        'val' stage. This may happen when the stage splits are
        performed inside the clouds. In this case, an
        `on_device_transform` will be automatically created to separate
        stage-specific data upon reading
    :param custom_hash: str
        A user-chosen hash to be used for the dataset data directory.
        This will bypass the default behavior where the pre_transforms
        are used to generate a hash. It can be used, for instance, when
        one wants to instantiate a dataset with already-processed data,
        without knowing the exact config that was used to generate it
    :param in_memory: bool
        If True, the processed dataset will be entirely loaded in RAM
        upon instantiation. This will accelerate training and inference
        but requires large memory. WARNING: __getitem__ directly
        returns the data in memory, so any modification to the returned
        object will affect the `in_memory_data` too. Be careful to clone
        the object before modifying it. Besides, the `transform` are
        pre-applied to the in_memory data
    :param point_save_keys: list[str]
        List of point (ie level-0) attribute keys to save to disk at
        the end of preprocessing. Leaving to `None` will save all
        attributes by default
    :param point_no_save_keys: list[str]
        List of point (ie level-0) attribute keys to NOT save to disk at
        the end of preprocessing
    :param point_load_keys: list[str]
        List of point (ie level-0) attribute keys to load when reading
        data from disk
    :param segment_save_keys: list[str]
        List of segment (ie level-1+) attribute keys to save to disk
        at the end of preprocessing. Leaving to `None` will save all
        attributes by default
    :param segment_no_save_keys: list[str]
        List of segment (ie level-1+) attribute keys to NOT save to disk
        at the end of preprocessing
    :param segment_load_keys: list[str]
        List of segment (ie level-1+) attribute keys to load when
        reading data from disk
    :param fold : int
        Integer in [1, ..., 6] indicating the Test Area
    :param with_stuff: bool
        By default, S3DIS does not have any stuff class. If `with_stuff`
        is True, the 'ceiling', 'wall', and 'floor' classes will be
        treated as stuff
    :param align : bool
        Whether the rooms should be canonically aligned, as described in
        section 3.2 of the S3DIS paper
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

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a `Data` object, ready to
        be passed to `self.pre_transform`.

        This `Data` object should contain the following attributes:
          - `pos`: point coordinates
          - `y`: OPTIONAL point semantic label
          - `obj`: OPTIONAL `InstanceData` object with instance labels
          - `rgb`: OPTIONAL point color
          - `intensity`: OPTIONAL point LiDAR intensity

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.
        """
        return read_s3dis_room(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=True,
            xyz_room=True, align=self.align, is_val=True, verbose=False)

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        room_folders = self.raw_file_names_3d
        area_folders = [f'Area_{i + 1}' for i in range(6)]
        alignment_files = [
            osp.join(a, f"{a}_alignmentAngle.txt") for a in area_folders]
        return room_folders + alignment_files

    def processed_to_raw_path(self, processed_path):
        """Given a processed cloud path from `self.processed_paths`,
        return the absolute path to the corresponding raw cloud.

        Overwrite this method if your raw data does not follow the
        default structure.
        """
        # Extract useful information from <path>
        stage, hash_dir, area_id, room_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-4:]
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
