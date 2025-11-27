import os
import sys
import glob
import torch
import shutil
import logging
import pandas as pd
import os.path as osp
from typing import List
from torch_geometric.data import extract_zip

from src.datasets import BaseDataset
from src.data import Data, Batch, InstanceData
from src.datasets.s3dis_config import *
from src.utils import (
    rodrigues_rotation_matrix,
    to_float_rgb)
from src.transforms import RoomPosition


DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['S3DIS', 'MiniS3DIS']


########################################################################
#                                 Utils                                #
########################################################################

def read_s3dis_area(
        area_dir: str,
        xyz: bool = True,
        rgb: bool = True,
        semantic: bool = True,
        instance: bool = True,
        xyz_room: bool = False,
        align: bool = False,
        is_val: bool = True,
        verbose: bool = False
) -> Data:
    """Read all S3DIS object-wise annotations in a given Area directory.
    All room-wise data are accumulated into a single cloud.

    :param area_dir: str
        Absolute path to the Area directory, eg: '/some/path/Area_1'
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param xyz_room: bool
        Whether the canonical room coordinates should be saved in the
        output Data.pos_room, as defined in the S3DIS paper section 3.2:
        https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf
    :param align: bool
        Whether the room should be rotated to its canonical orientation,
        as defined in the S3DIS paper section 3.2:
        https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf
    :param is_val: bool
        Whether the output `Batch.is_val` should carry a boolean label
        indicating whether they belong to the Area validation split
    :param verbose: bool
        Verbosity
    :return:
        Batch of accumulated points clouds
    """
    # List the object-wise annotation files in the room
    room_directories = sorted(
        [x for x in glob.glob(osp.join(area_dir, '*')) if osp.isdir(x)])

    # Read all rooms in the Area and concatenate point clouds in a Batch
    data_list = []
    for room in room_directories:
        data_list.append(read_s3dis_room(
            room,
            xyz=xyz,
            rgb=rgb,
            semantic=semantic,
            instance=instance,
            xyz_room=xyz_room,
            align=align,
            is_val=is_val,
            verbose=verbose))
    batch = Batch.from_data_list(data_list)

    # Convert from Batch to Data. Little hacky. We do this because we do
    # not want to store the Batch object but just a big Data object here
    # Why not store the Batch ? Because some pretransforms (e.g.
    # GridSampling3D) will want to treat the batch in a special way, but
    # we want the preprocessing to be agnostic to the room subdivision
    data = batch.forget_batching()

    return data


def read_s3dis_room(
        room_dir: str,
        xyz: bool = True,
        rgb: bool = True,
        semantic: bool = True,
        instance: bool = True,
        xyz_room: bool = False,
        align: bool = False,
        is_val: bool = True,
        verbose: bool = False
) -> Data:
    """Read all S3DIS object-wise annotations in a given room directory.

    :param room_dir: str
        Absolute path to the room directory, eg:
        '/some/path/Area_1/office_1'
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output `Data.pos`
    :param rgb: bool
        Whether RGB colors should be saved in the output `Data.rgb`
    :param semantic: bool
        Whether semantic labels should be saved in the output `Data.y`
    :param instance: bool
        Whether instance labels should be saved in the output `Data.obj`
    :param xyz_room: bool
        Whether the canonical room coordinates should be saved in the
        output Data.pos_room, as defined in the S3DIS paper section 3.2:
        https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf
    :param align: bool
        Whether the room should be rotated to its canonical orientation,
        as defined in the S3DIS paper section 3.2:
        https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf
    :param is_val: bool
        Whether the output `Data.is_val` should carry a boolean label
        indicating whether they belong to their Area validation split
    :param verbose: bool
        Verbosity

    :return: Data
    """
    if verbose:
        log.debug(f"Reading room: {room_dir}")

    # Initialize accumulators for xyz, RGB, semantic label and instance
    # label
    xyz_list = [] if xyz else None
    rgb_list = [] if rgb else None
    y_list = [] if semantic else None
    obj_list = [] if instance else None

    # List the object-wise annotation files in the room
    objects = sorted(glob.glob(osp.join(room_dir, 'Annotations', '*.txt')))

    # 'Area_5/office_36' contains two 'wall_3' annotation files, so we
    # manually remove the unwanted one
    objects = [
        p for p in objects
        if not p.endswith("Area_5/office_36/Annotations/wall_3 (1).txt")]

    for i_object, path in enumerate(objects):
        object_name = osp.splitext(osp.basename(path))[0]
        if verbose:
            log.debug(f"Reading object {i_object}: {object_name}")

        # Remove the trailing number in the object name to isolate the
        # object class (e.g. 'chair_24' -> 'chair')
        object_class = object_name.split('_')[0]

        # Convert object class string to int label. Note that by default
        # if an unknown class is read, it will be treated as 'clutter'.
        # This is necessary because an unknown 'staris' class can be
        # found in some rooms
        label = OBJECT_LABEL.get(object_class, OBJECT_LABEL['clutter'])
        points = pd.read_csv(path, sep=' ', header=None).values

        if xyz:
            xyz_list.append(
                np.ascontiguousarray(points[:, 0:3], dtype='float32'))

        if rgb:
            try:
                rgb_list.append(
                    np.ascontiguousarray(points[:, 3:6], dtype='uint8'))
            except ValueError:
                rgb_list.append(np.zeros((points.shape[0], 3), dtype='uint8'))
                log.warning(f"WARN - corrupted rgb data for file {path}")

        if semantic:
            y_list.append(np.full(points.shape[0], label, dtype='int64'))

        if instance:
            obj_list.append(np.full(points.shape[0], i_object, dtype='int64'))

    # Concatenate and convert to torch
    xyz_data = torch.from_numpy(np.concatenate(xyz_list, 0)) if xyz else None
    rgb_data = to_float_rgb(torch.from_numpy(np.concatenate(rgb_list, 0))) \
        if rgb else None
    y_data = torch.from_numpy(np.concatenate(y_list, 0)) if semantic else None

    # Store into a Data object
    pos_offset = torch.zeros_like(xyz_data[0]) if xyz else None
    data = Data(pos=xyz_data, pos_offset=pos_offset, rgb=rgb_data, y=y_data)

    # Store instance labels in InstanceData format
    if instance:
        idx = torch.arange(data.num_points)
        obj = torch.from_numpy(np.concatenate(obj_list, 0))
        count = torch.ones_like(obj)
        y = torch.from_numpy(np.concatenate(y_list, 0))
        data.obj = InstanceData(idx, obj, count, y, dense=True)

    # Add is_val attribute if need be
    if is_val:
        data.is_val = torch.ones(data.num_nodes, dtype=torch.bool) * (
                osp.basename(room_dir) in VALIDATION_ROOMS)

    # Exit here if canonical orientations are not needed
    if not xyz_room and not align:
        return data

    # Recover the canonical rotation angle for the room at hand. NB:
    # this assumes the raw files are stored in the S3DIS structure:
    #   raw/
    #     └── Area_{{i_area: 1 > 6}}/
    #       └── Area_{{i_area: 1 > 6}}_alignmentAngle.txt
    #       └── {{room_dir}}
    #       └── ...
    area_dir = osp.dirname(room_dir)
    area = osp.basename(osp.dirname(room_dir))
    room_name = osp.basename(room_dir)
    alignment_file = osp.join(area_dir, f'{area}_alignmentAngle.txt')
    alignments = pd.read_csv(
        alignment_file, sep=' ', header=None, skiprows=2).values
    angle = float(alignments[np.where(alignments[:, 0] == room_name), 1])

    # Matrix to rotate the room to its canonical orientation
    R = rodrigues_rotation_matrix(
        torch.tensor([0, 0, 1], dtype=torch.float),
        angle)

    # Rotate the room
    pos_bkp = data.pos
    data.pos = data.pos @ R

    # Save the required attributes
    if xyz_room:
        data = RoomPosition()(data)
    if not align:
        data.pos = pos_bkp

    return data


########################################################################
#                               S3DIS                               #
########################################################################

class S3DIS(BaseDataset):
    """S3DIS dataset, for Area-wise prediction.

    Note: we are using the S3DIS version with non-aligned rooms, which
    contains `Area_{{i_area:1>6}}_alignmentAngle.txt` files. Make sure
    you are not using the aligned version.

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
    """

    _form_url = FORM_URL
    _zip_name = ZIP_NAME
    _aligned_zip_name = ALIGNED_ZIP_NAME
    _unzip_name = UNZIP_NAME

    def __init__(
            self,
            *args,
            fold: int = 5,
            with_stuff: bool = False,
            **kwargs):
        self.fold = fold
        self.with_stuff = with_stuff
        super().__init__(*args, val_mixed_in_train=True, **kwargs)

    @property
    def pre_transform_hash(self) -> str:
        """Produce a unique but stable hash based on the dataset's
        `pre_transform` attributes (as exposed by `_repr`).

        For S3DIS, we want the hash to detect if the stuff classes are
        the default ones.
        """
        suffix = '_with_stuff' if self.with_stuff else ''
        return super().pre_transform_hash + suffix

    @property
    def class_names(self) -> List[str]:
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return S3DIS_NUM_CLASSES

    @property
    def stuff_classes(self) -> List[int]:
        """List of 'stuff' labels for INSTANCE and PANOPTIC
        SEGMENTATION (setting this is NOT REQUIRED FOR SEMANTIC
        SEGMENTATION alone). By definition, 'stuff' labels are labels in
        `[0, self.num_classes-1]` which are not 'thing' labels.

        In instance segmentation, 'stuff' classes are not taken into
        account in performance metrics computation.

        In panoptic segmentation, 'stuff' classes are taken into account
        in performance metrics computation. Besides, each cloud/scene
        can only have at most one instance of each 'stuff' class.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return STUFF_CLASSES_MODIFIED if self.with_stuff else STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> List[str]:
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return {
            'train': [f'Area_{i}' for i in range(1, 7) if i != self.fold],
            'val': [f'Area_{i}' for i in range(1, 7) if i != self.fold],
            'test': [f'Area_{self.fold}']}

    def download_dataset(self) -> None:
        """Download the S3DIS dataset.
        """
        # Manually download the dataset
        if not osp.exists(osp.join(self.root, self._zip_name)):
            log.error(
                f"\nS3DIS does not support automatic download.\n"
                f"Please, register yourself by filling up the form at "
                f"{self._form_url}\n"
                f"From there, manually download the non-aligned rooms"
                f"'{self._zip_name}' into your '{self.root}/' directory and "
                f"re-run.\n"
                f"The dataset will automatically be unzipped into the "
                f"following structure:\n"
                f"{self.raw_file_structure}\n"
                f"⛔ Make sure you DO NOT download the "
                f"'{self._aligned_zip_name}' version, which does not contain "
                f"the required `Area_{{i_area:1>6}}_alignmentAngle.txt` files."
                f"\n")
            sys.exit(1)

        # Unzip the file and rename it into the `root/raw/` directory. This
        # directory contains the raw Area folders from the zip
        extract_zip(osp.join(self.root, self._zip_name), self.root)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self._unzip_name), self.raw_dir)

    def read_single_raw_cloud(self, raw_cloud_path: str) -> 'Data':
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
        return read_s3dis_area(
            raw_cloud_path,
            xyz=True,
            rgb=True,
            semantic=True,
            instance=True,
            xyz_room=True,
            align=False,
            is_val=True,
            verbose=False)

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── {self._zip_name}
        └── raw/
            └── Area_{{i_area:1>6}}/
                └── Area_{{i_area:1>6}}_alignmentAngle.txt
                └── ...
            """

    @property
    def raw_file_names(self) -> str:
        """The file paths to find in order to skip the download."""
        area_folders = super().raw_file_names
        alignment_files = [
            osp.join(a, f"{a}_alignmentAngle.txt") for a in area_folders]
        return area_folders + alignment_files

    def id_to_relative_raw_path(self, id: str) -> str:
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id)


########################################################################
#                              MiniS3DIS                               #
########################################################################

class MiniS3DIS(S3DIS):
    """A mini version of S3DIS with only 1 area per stage for
    experimentation.
    """
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self) -> List[str]:
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self) -> str:
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self) -> None:
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self) -> None:
        super().download()
