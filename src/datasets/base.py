import os
import re
import sys
import os.path as osp
import torch
import random
import logging
import hashlib
import warnings
from tqdm import tqdm
from datetime import datetime
from itertools import product
from tqdm.auto import tqdm as tq
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data.dataset import _repr
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.data import NAG
from src.transforms import NAGSelectByKey, NAGRemoveKeys, SampleXYTiling, \
    SampleRecursiveMainXYAxisTiling
from src.visualization import show

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['BaseDataset']


########################################################################
#                             BaseDataset                              #
########################################################################

class BaseDataset(InMemoryDataset):
    """Base class for datasets.

    Child classes must overwrite the following methods (see respective
    docstrings for more details):

    ```
    MyDataset(BaseDataset):

        def class_names(self):
            pass

        def num_classes(self):
            pass

        def stuff_classes(self):
            pass

        def class_colors(self):
            # Optional: only if you want to customize your color palette
            # for visualization
            pass

        def all_base_cloud_ids(self):
            pass

        def download_dataset(self):
            pass

        def read_single_raw_cloud(self):
            pass

        def raw_file_structure(self):
            # Optional: only if your raw or processed file structure
            # differs from the default
            pass

        def id_to_relative_raw_path(self):
            # Optional: only if your raw or processed file structure
            # differs from the default
            pass

        def processed_to_raw_path(self):
            # Optional: only if your raw or processed file structure
            # differs from the default
            pass
    ```


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
    val_mixed_in_train: bool, optional
        whether the 'val' stage data is saved in the same clouds as the
        'train' stage. This may happen when the stage splits are
        performed inside the clouds. In this case, an
        `on_device_transform` will be automatically created to separate
        stage-specific data upon reading
    test_mixed_in_val: bool, optional
        whether the 'test' stage data is saved in the same clouds as the
        'val' stage. This may happen when the stage splits are
        performed inside the clouds. In this case, an
        `on_device_transform` will be automatically created to separate
        stage-specific data upon reading
    custom_hash: str, optional
        A user-chosen hash to be used for the dataset data directory.
        This will bypass the default behavior where the pre_transforms
        are used to generate a hash. It can be used, for instance, when
        one wants to instantiate a dataset with already-processed data,
        without knowing the exact config that was used to generate it
    in_memory: bool, optional
        If True, the processed dataset will be entirely loaded in RAM
        upon instantiation. This will accelerate training and inference
        but requires large memory. WARNING: __getitem__ directly
        returns the data in memory, so any modification to the returned
        object will affect the `in_memory_data` too. Be careful to clone
        the object before modifying it. Besides, the `transform` are
        pre-applied to the in_memory data
    point_save_keys: list[str], optional
        List of point (ie level-0) attribute keys to save to disk at 
        the end of preprocessing. Leaving to `None` will save all 
        attributes by default
    point_no_save_keys: list[str], optional
        List of point (ie level-0) attribute keys to NOT save to disk at
        the end of preprocessing
    point_load_keys: list[str], optional
        List of point (ie level-0) attribute keys to load when reading 
        data from disk
    segment_save_keys: list[str], optional
        List of segment (ie level-1+) attribute keys to save to disk 
        at the end of preprocessing. Leaving to `None` will save all 
        attributes by default
    segment_no_save_keys: list[str], optional
        List of segment (ie level-1+) attribute keys to NOT save to disk 
        at the end of preprocessing
    segment_load_keys: list[str], optional
        List of segment (ie level-1+) attribute keys to load when 
        reading data from disk 
    """

    def __init__(
            self,
            root,
            stage='train',
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
            **kwargs):

        assert stage in ['train', 'val', 'trainval', 'test']

        # Set these attributes before calling parent `__init__` because
        # some attributes will be needed in parent `download` and
        # `process` methods
        self._stage = stage
        self._save_y_to_csr = save_y_to_csr
        self._save_pos_dtype = save_pos_dtype
        self._save_fp_dtype = save_fp_dtype
        self._on_device_transform = on_device_transform
        self._val_mixed_in_train = val_mixed_in_train
        self._test_mixed_in_val = test_mixed_in_val
        self._custom_hash = custom_hash
        self._in_memory = in_memory
        self._point_save_keys = point_save_keys
        self._point_no_save_keys = point_no_save_keys
        self._point_load_keys = point_load_keys
        self._segment_save_keys = segment_save_keys
        self._segment_no_save_keys = segment_no_save_keys
        self._segment_load_keys = segment_load_keys

        if in_memory:
            log.warning(
                "'in_memory' was set to True. This means the entire dataset "
                "will be held in RAM. While this allows training and inference "
                "speedups, this means that the `transform' will only be "
                "applied once, upon loading the dataset to RAM. Hence, if you "
                "need augmentations or any other stochastic operations to be "
                "applied on your batches, make sure you moved them all to "
                "'on_device_transform'.")

        # Prepare tiling arguments. Can either be XY tiling of PC
        # tiling but not both. XY tiling will apply a regular grid along
        # the XY axes to the data, regardless of its orientation, shape
        # or density. The value of xy_tiling indicates the number of
        # tiles in each direction. So, if a single int is passed, each
        # cloud will be divided into xy_tiling**2 tiles. PC tiling will
        # recursively split the data wrt the principal component along
        # the XY plane. Each step splits the data in 2, wrt to its
        # geometry. The value of pc_tiling indicates the number of split
        # steps used. Hence, 2**pc_tiling tiles will be created.
        assert xy_tiling is None or pc_tiling is None, \
            "Cannot apply both XY and PC tiling, please choose only one."
        if xy_tiling is None:
            self.xy_tiling = None
        elif isinstance(xy_tiling, int):
            self.xy_tiling = (xy_tiling, xy_tiling) if xy_tiling > 1 else None
        elif xy_tiling[0] > 1 or xy_tiling[1] > 1:
            self.xy_tiling = xy_tiling
        else:
            self.xy_tiling = None
        self.pc_tiling = pc_tiling if pc_tiling and pc_tiling >= 1 else None

        # Sanity check on the cloud ids. Ensures cloud ids are unique
        # across all stages, unless `val_mixed_in_train` or
        # `test_mixed_in_val` is True
        self.check_cloud_ids()

        # Initialization with downloading and all preprocessing
        root = osp.join(root, self.data_subdir_name)
        super().__init__(root, transform, pre_transform, pre_filter)

        # Display the dataset pre_transform_hash and full path
        path = osp.join(self.processed_dir, "<stage>", self.pre_transform_hash)
        log.info(f'Dataset hash: "{self.pre_transform_hash}"')
        log.info(f'Preprocessed data can be found at: "{path}"')

        # If `val_mixed_in_train` or `test_mixed_in_val`, we will need
        # to separate some stage-related data at reading time.
        # Since this operation can be computationally-costly, we prefer
        # postponing it to the `on_device_transform`. To this end, we
        # prepend the adequate transform to the dataset's
        # `on_device_transform`. Otherwise, if we have no mixed-stages,
        # we simply remove all `is_val` attributes in the
        # `on_device_transform`
        if self.stage == 'train' and self.val_mixed_in_train:
            t = NAGSelectByKey(key='is_val', negation=True)
        elif self.stage == 'val' and self.val_mixed_in_train or self.test_mixed_in_val:
            t = NAGSelectByKey(key='is_val', negation=False)
        elif self.stage == 'test' and self.test_mixed_in_val:
            t = NAGSelectByKey(key='is_val', negation=True)
        else:
            t = NAGRemoveKeys(level='all', keys=['is_val'], strict=False)

        # Make sure a NAGRemoveKeys for `is_val` does not already exist
        # in the `on_device_transform` before prepending the transform
        if not any(
                isinstance(odt, NAGSelectByKey) and odt.key == 'is_val'
                for odt in self.on_device_transform.transforms):
            self._on_device_transform.transforms = \
                [t] + self._on_device_transform.transforms

        # Load the processed data, if the dataset must be in memory
        if self.in_memory:
            in_memory_data = [
                NAG.load(
                    self.processed_paths[i],
                    keys_low=self.point_load_keys,
                    keys=self.segment_load_keys)
                for i in range(len(self))]
            if self.transform is not None:
                in_memory_data = [self.transform(x) for x in in_memory_data]
            self._in_memory_data = in_memory_data
        else:
            self._in_memory_data = None

    @property
    def class_names(self):
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        raise NotImplementedError

    @property
    def num_classes(self):
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def thing_classes(self):
        """List of 'thing' labels for instance and panoptic
        segmentation. By definition, 'thing' labels are labels in
        `[0, self.num_classes-1]` which are not 'stuff' labels.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return [i for i in range(self.num_classes) if i not in self.stuff_classes]

    @property
    def void_classes(self):
        """List containing the 'void' labels. By default, we group all
        void/ignored/unknown class labels into a single
        `[self.num_classes]` label for simplicity.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return [self.num_classes]

    @property
    def class_colors(self):
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return

    def print_classes(self):
        """Show the class names, labels and type (thing, stuff, void).
        """
        for i, c in enumerate(self.class_names):
            try:
                class_type = \
                    'stuff' if i in self.stuff_classes \
                    else 'thing' if i in self.thing_classes \
                    else 'void'
            except:
                class_type = ''
            print(f"{i:<3} {c:<20} {class_type}")

    @property
    def data_subdir_name(self):
        return self.__class__.__name__.lower()

    @property
    def stage(self):
        """Dataset stage. Expected to be 'train', 'val', 'trainval',
        or 'test'
        """
        return self._stage
    
    @property
    def save_y_to_csr(self):
        return self._save_y_to_csr

    @property
    def save_pos_dtype(self):
        return self._save_pos_dtype

    @property
    def save_fp_dtype(self):
        return self._save_fp_dtype

    @property
    def on_device_transform(self):
        return self._on_device_transform

    @property
    def val_mixed_in_train(self):
        return self._val_mixed_in_train

    @property
    def test_mixed_in_val(self):
        return self._test_mixed_in_val

    @property
    def custom_hash(self):
        return self._custom_hash

    @property
    def in_memory(self):
        return self._in_memory

    @property
    def point_save_keys(self):
        return self._point_save_keys

    @property
    def point_no_save_keys(self):
        return self._point_no_save_keys

    @property
    def point_load_keys(self):
        return self._point_load_keys

    @property
    def segment_save_keys(self):
        return self._segment_save_keys

    @property
    def segment_no_save_keys(self):
        return self._segment_no_save_keys

    @property
    def segment_load_keys(self):
        return self._segment_load_keys
        
    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        raise NotImplementedError

    @property
    def all_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage. Unlike all_base_cloud_ids, these ids take into account
        the clouds tiling, if any.
        """
        # If clouds are tiled, expand and append all cloud names with a
        # suffix indicating which tile it corresponds to
        if self.xy_tiling is not None:
            tx, ty = self.xy_tiling
            return {
                stage: [
                    f'{ci}__TILE_{x + 1}-{y + 1}_OF_{tx}-{ty}'
                    for ci in ids
                    for x, y in product(range(tx), range(ty))]
                for stage, ids in self.all_base_cloud_ids.items()}

        if self.pc_tiling is not None:
            return {
                stage: [
                    f'{ci}__TILE_{x + 1}_OF_{2**self.pc_tiling}'
                    for ci in ids
                    for x in range(2**self.pc_tiling)]
                for stage, ids in self.all_base_cloud_ids.items()}

        # If no tiling needed, return the all_base_cloud_ids
        return self.all_base_cloud_ids

    def id_to_base_id(self, id):
        """Given an ID, remove the tiling indications, if any.
        """
        if self.xy_tiling is None and self.pc_tiling is None:
            return id
        return self.get_tile_from_path(id)[1]

    @property
    def cloud_ids(self):
        """IDs of the dataset clouds, based on its `stage`.
        """
        if self.stage == 'trainval':
           ids = self.all_cloud_ids['train'] + self.all_cloud_ids['val']
        else:
            ids = self.all_cloud_ids[self.stage]
        return sorted(list(set(ids)))

    def check_cloud_ids(self):
        """Make sure the `all_cloud_ids` are valid. More specifically,
        the cloud ids must be unique across all stages, unless
        `val_mixed_in_train=True` or `test_mixed_in_val=True`, in
        which case some clouds may appear in several stages
        """
        train = set(self.all_cloud_ids['train'])
        val = set(self.all_cloud_ids['val'])
        test = set(self.all_cloud_ids['test'])

        assert len(train.intersection(val)) == 0 or self.val_mixed_in_train, \
            "Cloud ids must be unique across all the 'train' and 'val' " \
            "stages, unless `val_mixed_in_train=True`"
        assert len(val.intersection(test)) == 0 or self.test_mixed_in_val, \
            "Cloud ids must be unique across all the 'val' and 'test' " \
            "stages, unless `test_mixed_in_val=True`"

    @property
    def raw_file_structure(self):
        """String to describe to the user the file structure of your
        dataset, at download time.
        """
        return

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        return self.raw_file_names_3d

    @property
    def raw_file_names_3d(self):
        """Some file paths to find in order to skip the download.
        Those are not directly specified inside `self.raw_file_names`
        in case `self.raw_file_names` would need to be extended (e.g.
        with 3D bounding boxes files).
        """
        return [self.id_to_relative_raw_path(x) for x in self.cloud_ids]

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id) + '.ply'

    @property
    def pre_transform_hash(self):
        """Produce a unique but stable hash based on the dataset's
        `pre_transform` attributes (as exposed by `_repr`).
        """
        if self.custom_hash is not None:
            return self.custom_hash
        if self.pre_transform is None:
            return 'no_pre_transform'
        return hashlib.md5(_repr(self.pre_transform).encode()).hexdigest()

    @property
    def processed_file_names(self):
        """The name of the files to find in the `self.processed_dir`
        folder in order to skip the processing
        """
        # For 'trainval', we use files from 'train' and 'val' to save
        # memory
        if self.stage == 'trainval' and self.val_mixed_in_train:
            return [
                osp.join('train', self.pre_transform_hash, f'{w}.h5')
                for s in ('train', 'val')
                for w in self.all_cloud_ids[s]]
        if self.stage == 'trainval':
            return [
                osp.join(s, self.pre_transform_hash, f'{w}.h5')
                for s in ('train', 'val')
                for w in self.all_cloud_ids[s]]
        return [
            osp.join(self.stage, self.pre_transform_hash, f'{w}.h5')
            for w in self.cloud_ids]

    def processed_to_raw_path(self, processed_path):
        """Given a processed cloud path from `self.processed_paths`,
        return the absolute path to the corresponding raw cloud.

        Overwrite this method if your raw data does not follow the
        default structure.
        """
        # Extract useful information from <path>
        stage, hash_dir, cloud_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-3:]

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_ext = osp.splitext(self.raw_file_names_3d[0])[1]
        raw_path = osp.join(self.raw_dir, base_cloud_id + raw_ext)

        return raw_path

    @property
    def in_memory_data(self):
        """If the `self.in_memory`, this will return all processed data,
        loaded in memory. Returns None otherwise.
        """
        return self._in_memory_data

    @property
    def submission_dir(self):
        """Submissions are saved in the `submissions` folder, in the
        same hierarchy as `raw` and `processed` directories. Each
        submission has a subdirectory of its own, named based on the
        date and time of creation.
        """
        submissions_dir = osp.join(self.root, "submissions")
        date = '-'.join([
            f'{getattr(datetime.now(), x)}'
            for x in ['year', 'month', 'day']])
        time = '-'.join([
            f'{getattr(datetime.now(), x)}'
            for x in ['hour', 'minute', 'second']])
        submission_name = f'{date}_{time}'
        path = osp.join(submissions_dir, submission_name)
        return path

    def download(self):
        self.download_warning()
        self.download_dataset()

    def download_dataset(self):
        """Download the dataset data. Modify this method to implement
        your own `BaseDataset` child class.
        """
        raise NotImplementedError

    def download_warning(self, interactive=False):
        # Warning message for the user about to download
        log.info(
            f"WARNING: You must download the raw data for the "
            f"{self.__class__.__name__} dataset.")
        if self.raw_file_structure is not None:
            log.info("Files must be organized in the following structure:")
            log.info(self.raw_file_structure)
        log.info("")
        if interactive:
            log.info("Press any key to continue, or CTRL-C to exit.")
            input("")
            log.info("")

    def download_message(self, msg):
        log.info(f'Downloading "{msg}" to {self.raw_dir}...')

    def _process(self):
        """Overwrites torch-geometric's Dataset._process. This simply
        removes the 'pre_transform.pt' file used for checking whether
        the pre-transforms have changed. This is possible thanks to our
        `pre_transform_hash` mechanism.
        """
        f = osp.join(self.processed_dir, 'pre_filter.pt')
        if osp.exists(f) and torch.load(f) != _repr(self.pre_filter):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-filtering technique, make sure to "
                "delete '{self.processed_dir}' first")

        if files_exist(self.processed_paths):  # pragma: no cover
            return

        if self.log and 'pytest' not in sys.modules:
            print('Processing...', file=sys.stderr)

        makedirs(self.processed_dir)
        self.process()

        path = osp.join(self.processed_dir, 'pre_filter.pt')
        torch.save(_repr(self.pre_filter), path)

        if self.log and 'pytest' not in sys.modules:
            print('Done!', file=sys.stderr)

    def process(self):
        # If some stages have mixed clouds (they rely on the same cloud
        # files and the split is operated at reading time by
        # `on_device_transform`), we create symlinks between the
        # necessary folders, to avoid duplicate preprocessing
        # computation
        hash_dir = self.pre_transform_hash
        train_dir = osp.join(self.processed_dir, 'train', hash_dir)
        val_dir = osp.join(self.processed_dir, 'val', hash_dir)
        test_dir = osp.join(self.processed_dir, 'test', hash_dir)
        if not osp.exists(train_dir):
            os.makedirs(train_dir, exist_ok=True)
        if not osp.exists(val_dir):
            if self.val_mixed_in_train:
                os.makedirs(osp.dirname(val_dir), exist_ok=True)
                os.symlink(train_dir, val_dir, target_is_directory=True)
            else:
                os.makedirs(val_dir, exist_ok=True)
        if not osp.exists(test_dir):
            if self.test_mixed_in_val:
                os.makedirs(osp.dirname(test_dir), exist_ok=True)
                os.symlink(val_dir, test_dir, target_is_directory=True)
            else:
                os.makedirs(test_dir, exist_ok=True)

        # Process clouds one by one
        for p in tq(self.processed_paths):
            self._process_single_cloud(p)

    def _process_single_cloud(self, cloud_path):
        """Internal method called by `self.process` to preprocess a
        single cloud of 3D points.
        """
        # If required files exist, skip processing
        if osp.exists(cloud_path):
            return

        # Create necessary parent folders if need be
        os.makedirs(osp.dirname(cloud_path), exist_ok=True)

        # Read the raw cloud corresponding to the final processed
        # `cloud_path` and convert it to a Data object
        raw_path = self.processed_to_raw_path(cloud_path)
        data = self.sanitized_read_single_raw_cloud(raw_path)

        # If the cloud path indicates a tiling is needed, apply it here
        if self.xy_tiling is not None:
            tile = self.get_tile_from_path(cloud_path)[0]
            data = SampleXYTiling(x=tile[0], y=tile[1], tiling=tile[2])(data)
        elif self.pc_tiling is not None:
            tile = self.get_tile_from_path(cloud_path)[0]
            data = SampleRecursiveMainXYAxisTiling(x=tile[0], steps=tile[1])(data)

        # Apply pre_transform
        if self.pre_transform is not None:
            nag = self.pre_transform(data)
        else:
            nag = NAG([data])

        # To save some disk space, we discard some level-0 attributes
        if self.point_save_keys is not None:
            keys = set(nag[0].keys) - set(self.point_save_keys)
            nag = NAGRemoveKeys(level=0, keys=keys)(nag)
        elif self.point_no_save_keys is not None:
            nag = NAGRemoveKeys(level=0, keys=self.point_no_save_keys)(nag)
        if self.segment_save_keys is not None:
            keys = set(nag[1].keys) - set(self.segment_save_keys)
            nag = NAGRemoveKeys(level='1+', keys=keys)(nag)
        elif self.segment_no_save_keys is not None:
            nag = NAGRemoveKeys(level=0, keys=self.segment_no_save_keys)(nag)

        # Save pre_transformed data to the processed dir/<path>
        # TODO: is you do not throw away level-0 neighbors, make sure
        #  that they contain no '-1' empty neighborhoods, because if
        #  you load them for batching, the pyg reindexing mechanism will
        #  break indices will not index update
        nag.save(
            cloud_path,
            y_to_csr=self.save_y_to_csr,
            pos_dtype=self.save_pos_dtype,
            fp_dtype=self.save_fp_dtype)
        del nag

    @staticmethod
    def get_tile_from_path(path):
        # Search the XY tiling suffix pattern
        out_reg = re.search('__TILE_(\d+)-(\d+)_OF_(\d+)-(\d+)', path)
        if out_reg is not None:
            x, y, x_tiling, y_tiling = [int(g) for g in out_reg.groups()]
            suffix = f'__TILE_{x}-{y}_OF_{x_tiling}-{y_tiling}'
            prefix = path.replace(suffix, '')
            return (x - 1, y - 1, (x_tiling, y_tiling)), prefix, suffix

        # Search the PC tiling suffix pattern
        out_reg = re.search('__TILE_(\d+)_OF_(\d+)', path)
        if out_reg is not None:
            x, num = [int(g) for g in out_reg.groups()]
            suffix = f'__TILE_{x}_OF_{num}'
            prefix = path.replace(suffix, '')
            steps = torch.log2(torch.tensor(num)).int().item()
            return (x - 1, steps), prefix, suffix

        return

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
        raise NotImplementedError

    def sanitized_read_single_raw_cloud(self, raw_cloud_path):
        """Wrapper around the actual `self.read_single_raw_cloud`. This
        function ensures that the semantic and instance segmentation
        labels returned by the reader are sanitized.

        More specifically, we assume `[0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.

        To this end, this function maps all labels outside
        `[0, self.num_classes-1]` to `y = self.num_classes`.

        Hence, we actually have `self.num_classes + 1` labels in the
        data. This allows identifying the points to be ignored at metric
        computation time.

        Besides, this function ensures that there is at most 1 instance
        of each stuff (and void) class in each scene/cloud/tile, as
        described in:
          - https://arxiv.org/abs/1801.00868
          - https://arxiv.org/abs/1905.01220
        """
        data = self.read_single_raw_cloud(raw_cloud_path)

        # Set all void labels to self.num_classes in the semantic
        # segmentation labels
        if getattr(data, 'y', None) is not None:
            data.y[data.y < 0] = self.num_classes
            data.y[data.y > self.num_classes] = self.num_classes

        # Set all void labels to self.num_classes in the
        # instance/panoptic segmentation annotations
        if getattr(data, 'obj', None) is not None:
            data.obj.y[data.obj.y < 0] = self.num_classes
            data.obj.y[data.obj.y > self.num_classes] = self.num_classes

            # For each cloud/scene and each stuff/void class, group
            # annotations into a single instance
            for i in self.stuff_classes + self.void_classes:
                idx = torch.where(data.obj.y == i)[0]
                if idx.numel() == 0:
                    continue
                data.obj.obj[idx] = data.obj.obj[idx].min()

        return data

    def debug_instance_data(self, level=1):
        """Sanity check to make sure at most 1 instance of each stuff
        class per scene/cloud.

        :param level: int
            NAG level which to inspect
        """
        problematic_clouds = []
        for i_cloud, nag in tqdm(enumerate(self)):
            _, perm = consecutive_cluster(nag[level].obj.obj)
            y = nag[level].obj.y[perm]
            y_count = torch.bincount(y, minlength=self.num_classes + 1)
            for c in self.stuff_classes + self.void_classes:
                if y_count[c] > 1:
                    problematic_clouds.append(i_cloud)
                    break

        assert len(problematic_clouds) == 0, \
            f"The following clouds have more than 1 instance of for a stuff " \
            f"or void class:\n{problematic_clouds}"

    def get_class_weight(self, smooth='sqrt'):
        """Compute class weights based on the labels distribution in the
        dataset. Optionally a 'smooth' function may be passed to
        smoothen the weights' statistics.
        """
        assert smooth in [None, 'sqrt', 'log']

        # Read the first NAG just to know how many levels we have in the
        # preprocessed NAGs.
        nag = self[0]
        low = nag.num_levels - 1

        # Make sure the dataset has labels
        if nag[low].y is None:
            return None
        del nag

        # To be as fast as possible, we read only the last level of each
        # NAG, and accumulate the class counts from the label histograms
        counts = torch.zeros(self.num_classes)
        for i in range(len(self)):
            if self.in_memory:
                y = self.in_memory_data[i][low].y
            else:
                y = NAG.load(
                    self.processed_paths[i], low=low, keys_low=['y'])[0].y
            counts += y.sum(dim=0)[:self.num_classes]

        # Compute the class weights. Optionally, a 'smooth' function may
        # be applied to smoothen the weights statistics
        if smooth == 'sqrt':
            counts = counts.sqrt()
        if smooth == 'log':
            counts = counts.log()

        weights = 1 / (counts + 1)
        weights /= weights.sum()

        return weights

    def __len__(self):
        """Number of clouds in the dataset."""
        return len(self.cloud_ids)

    def __getitem__(self, idx):
        """Load a preprocessed NAG from disk and apply `self.transform`
        if any. Optionally, one may pass a tuple (idx, bool) where the
        boolean indicates whether the data should be loaded from disk, if
        `self.in_memory=True`.
        """
        # Prepare from_hdd
        from_hdd = False
        if isinstance(idx, tuple):
            assert len(idx) == 2 and isinstance(idx[1], bool), \
                "Only supports indexing with `int` or `(int, bool)` where the" \
                " boolean indicates whether the data should be loaded from " \
                "disk, when `self.in_memory=True`."
            idx, from_hdd = idx

        # Get the processed NAG directly from RAM
        if self.in_memory and not from_hdd:
            # TODO: careful, this means the transforms are only run
            #  once. So no augmentations, samplings, etc in the
            #  transforms...
            return self.in_memory_data[idx]

        # Read the NAG from HDD
        nag = NAG.load(
            self.processed_paths[idx],
            keys_low=self.point_load_keys,
            keys=self.segment_load_keys)

        # Apply transforms
        nag = nag if self.transform is None else self.transform(nag)

        return nag

    def make_submission(self, idx, pred, pos, submission_dir=None):
        """Implement this if your dataset needs to produce data in a
        given format for submission. This is typically needed for
        datasets with held-out test sets.
        """
        raise NotImplementedError

    def finalize_submission(self, submission_dir):
        """Implement this if your dataset needs to produce data in a
        given format for submission. This is typically needed for
        datasets with held-out test sets.
        """
        raise NotImplementedError

    def show_examples(
            self, label, radius=4, max_examples=5, shuffle=True, **kwargs):
        """Interactive plots of some examples centered on points of the
        provided `label`. At most one example per cloud/tile/scene in
        the dataset will be shown.

        :param label: int or str
            Label of the class of interest, may be provided as an int or
            a string corresponding to the class name
        :param radius: float
            Radius of the spherical sampling to draw around the point of
            interest
        :param max_examples: int
            Maximum number of samples to draw
        :param shuffle: bool
            If True, the candidate samples will be shuffled every time
        :param kwargs:
            Kwargs to be passed to the visualization `show()` function
        :return:
        """
        if isinstance(label, str):
            assert label in self.class_names, \
                f"Label must be within {self.class_names}]"
            label = self.class_names.index(label)
        
        assert label >= 0 and label <= self.num_classes, \
            f"Label must be within [0, {self.num_classes + 1}]"

        # Gather some clouds ids with the desired class
        cloud_list = []
        iterator = list(range(len(self)))
        if shuffle:
            random.shuffle(iterator)
        for i_cloud in iterator:
            if len(cloud_list) >= max_examples:
                break
            if (self[i_cloud][1].y.argmax(dim=1) == label).any():
                cloud_list.append(i_cloud)

        # If no cloud was found with the desired class, return here
        if len(cloud_list) == 0:
            print(
                f"Could not find any cloud with points of label={label} in the "
                f"dataset.")
            return

        # Display some found examples
        for i, i_cloud in enumerate(cloud_list):
            if i >= max_examples:
                break

            # Load the cloud
            nag = self[i_cloud]

            # Search for points with the desired label
            point_idx = torch.where(nag[0].y.argmax(dim=1) == label)[0].tolist()

            # Pick only on of the points as visualization center for the
            # cloud at hand
            if shuffle:
                random.shuffle(point_idx)
            i_point = point_idx[0]

            # Draw the scene
            center = nag[0].pos[i_point].cpu().tolist()
            title = f"Label={label} - Cloud={i_cloud} - Center={center}"
            print(f"\n{title}")
            show(
                nag,
                center=center,
                radius=radius,
                title=title,
                class_names=self.class_names,
                class_colors=self.class_colors,
                stuff_classes=self.stuff_classes,
                num_classes=self.num_classes,
                **kwargs)
