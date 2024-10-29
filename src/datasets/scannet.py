import os
import sys
import torch
import logging
import os.path as osp
from src.data import Data, InstanceData
from src.utils.scannet import read_one_scan, read_one_test_scan
from src.datasets.scannet_config import *
from src.datasets.base import BaseDataset
from torch_geometric.nn.pool.consecutive import consecutive_cluster


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
        scan_dir,
        xyz=True,
        rgb=True,
        normal=True,
        semantic=True,
        instance=True,
        remap=True):
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

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    stage : {'train', 'val', 'test', 'trainval'}
    transform : `callable`
        transform function operating on data.
    pre_transform : `callable`
        pre_transform function operating on data.
    pre_filter : `callable`
        pre_filter function operating on data.
    on_device_transform: `callable`
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _form_url = FORM_URL

    @property
    def class_names(self):
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return SCANNET_NUM_CLASSES

    @property
    def stuff_classes(self):
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
    def class_colors(self):
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of paths to the clouds, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return SCANS

    def download_dataset(self):
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
        return read_scannet_scan(
            raw_cloud_path,
            xyz=True,
            rgb=True,
            normal=True,
            semantic=True,
            instance=True,
            remap=True)

    @property
    def raw_file_structure(self):
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

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id)

    def processed_to_raw_path(self, processed_path):
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
    def raw_file_names(self):
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
