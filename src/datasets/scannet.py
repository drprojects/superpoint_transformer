import os
import sys
import torch
import logging
import os.path as osp
from typing import List
from torch_geometric.nn.pool.consecutive import consecutive_cluster

from src.data import Data, InstanceData
from src.utils.scannet import read_one_scan, read_one_test_scan
from src.datasets.scannet_config import *
from src.datasets.base import BaseDataset


DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


# Occasional Dataloader issues on some machines. Hack to solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__all__ = ['ScanNet', 'MiniScanNet']


########################################################################
#                                 Utils                                #
########################################################################

def read_scannet_scan(
        scan_dir: str,
        xyz: bool = True,
        rgb: bool = True,
        normal: bool = True,
        semantic: bool = True,
        instance: bool = True,
        remap: bool = True
) -> Data:
    """Read a ScanNet scan.

    Expects the data to be saved under the following structure:

        └── raw/
            ├── scannetv2-labels.combined.tsv
            ├── scans/
            │   └── {{scan_name}}/
            │       ├── {{scan_name}}.aggregation.json
            │       ├── {{scan_name}}.txt
            │       ├── {{scan_name}}_vh_clean_2.0.010000.segs.json
            │       └── {{scan_name}}_vh_clean_2.ply
            └── scans_test/
                └── {{scan_name}}/
                    └── {{scan_name}}_vh_clean_2.ply

    :param scan_dir: str
        Absolute path to the directory
        `raw/{{scans, scans_test}}/{{scan_name}}/{{scan_name}}`
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param normal: bool
        Whether normals should be saved in the output Data.normal
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their NYU40 ID
        to their ScanNet ID
    """
    # Remove trailing slash, just in case
    scan_dir = scan_dir[:-1] if scan_dir[-1] == '/' else scan_dir

    # Extract the parent directory and the scan name from the scan
    # directory path. The parent directory will be used to identify test
    # scans
    scan_name = osp.basename(scan_dir)
    stage_dir = osp.dirname(scan_dir)
    stage_dirname = osp.basename(stage_dir)
    raw_dir = osp.dirname(stage_dir)

    # Build the path to the label mapping .tsv file, it is expected to
    # be in the raw/ folder
    label_map_file = osp.join(raw_dir, "scannetv2-labels.combined.tsv")

    # Scans are expected in the 'raw_dir/{scans, scans_test}/scan_name'
    # structure
    if stage_dirname not in ['scans', 'scans_test']:
        raise ValueError(
            "Expected the data to be in a "
            "'raw_dir/{scans, scans_test}/scan_name' structure, but parent "
            f"directory is {stage_dirname}")

    # Read the scan. Different reading methods for train/val scans and
    # test scans
    if stage_dirname == 'scans':
        pos, color, n, y, obj = read_one_scan(stage_dir, scan_name, label_map_file)
        y = torch.from_numpy(NYU40_2_SCANNET)[y] if remap else y
        pos_offset = torch.zeros_like(pos[0])
        data = Data(pos=pos, pos_offset=pos_offset, rgb=color, normal=n, y=y)
        idx = torch.arange(data.num_points)
        obj = consecutive_cluster(obj)[0]
        count = torch.ones_like(obj)
        data.obj = InstanceData(idx, obj, count, y, dense=True)
    else:
        pos, color, n = read_one_test_scan(stage_dir, scan_name)
        pos_offset = torch.zeros_like(pos[0])
        data = Data(pos=pos, pos_offset=pos_offset, rgb=color, normal=n)

    # Sometimes the returned normals may be 0. Since normals are assumed
    # to have unit-norm, and to avoid downstream errors, we arbitrarily
    # choose to set those problematic normals [0, 0, 1]
    idx = torch.where(n.norm(dim=1) == 0)[0]
    n[idx] = torch.tensor([0, 0, 1], dtype=torch.float)

    # Remove unneeded attributes
    if not xyz:
        data.pos = None
    if not rgb:
        data.rgb = None
    if not normal:
        data.normal = None
    if not semantic:
        data.y = None
    if not instance:
        data.obj = None

    return data


########################################################################
#                               ScanNet                                #
########################################################################

class ScanNet(BaseDataset):
    """ScanNet dataset.

    Dataset website: http://www.scan-net.org

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
    """

    _form_url = FORM_URL

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
        return SCANNET_NUM_CLASSES

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
        return STUFF_CLASSES

    @property
    def class_colors(self) -> List[List[int]]:
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self) -> List[str]:
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return SCANS

    def download_dataset(self) -> None:
        """Download the ScanNet dataset.
        """
        log.error(
            f"\nScanNet does not support automatic download.\n"
            f"Please go to the official webpage {self._form_url},"
            f"download the files indicated below into your {self.root}/' "
            f"directory, and re-run.\n"
            f"The dataset must be organized into the following structure:\n"
            f"{self.raw_file_structure}\n")
        sys.exit(1)

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
        return read_scannet_scan(
            raw_cloud_path,
            xyz=True,
            rgb=True,
            normal=True,
            semantic=True,
            instance=True,
            remap=True)

    @property
    def raw_file_structure(self) -> str:
        return f"""
    {self.root}/
        └── raw/
            ├── scannetv2-labels.combined.tsv
            ├── scans/
            │   └── {{scan_name}}/
            │       ├── {{scan_name}}.aggregation.json
            │       ├── {{scan_name}}.txt
            │       ├── {{scan_name}}_vh_clean_2.0.010000.segs.json
            │       └── {{scan_name}}_vh_clean_2.ply
            └── scans_test/
                └── {{scan_name}}/
                    └── {{scan_name}}_vh_clean_2.ply
            """

    def id_to_relative_raw_path(self, id: str) -> str:
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id)

    def processed_to_raw_path(self, processed_path: str) -> str:
        """Given a processed cloud path from `self.processed_paths`,
        return the absolute path to the corresponding raw cloud.

        Overwrite this method if your raw data does not follow the
        default structure.
        """
        # Extract useful information from <path>
        stage, hash_dir, scans_dir, scan_name = \
            osp.splitext(processed_path)[0].split(os.sep)[-4:]
        cloud_id = osp.join(scans_dir, scan_name)

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_ext = osp.splitext(self.raw_file_names_3d[0])[1]
        raw_path = osp.join(self.raw_dir, base_cloud_id + raw_ext)

        return raw_path

    @property
    def raw_file_names(self) -> str:
        """The file paths to find in order to skip the download."""
        area_folders = super().raw_file_names
        label_mapping_file = 'scannetv2-labels.combined.tsv'
        return area_folders + [label_mapping_file]


########################################################################
#                             MiniKITTI360                             #
########################################################################

class MiniScanNet(ScanNet):
    """A mini version of ScanNet with only a few scans for
    experimentation.
    """
    _NUM_MINI = 2

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
