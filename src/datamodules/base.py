import torch
import logging
from pytorch_lightning import LightningDataModule
from src.transforms import *
from src.loader import DataLoader
from src.data import NAGBatch


log = logging.getLogger(__name__)


# List of transforms not allowed for test-time augmentation
_TTA_CONFLICTS = []

# List of transforms not allowed for test prediction submission
_SUBMISSION_CONFLICTS = [
    CenterPosition,
    RandomTiltAndRotate,
    RandomAnisotropicScale,
    RandomAxisFlip,
    Inliers,
    Outliers,
    Shuffle,
    GridSampling3D,
    SampleXYTiling,
    SampleRecursiveMainXYAxisTiling,
    SampleSubNodes,
    SampleSegments,
    SampleKHopSubgraphs,
    SampleRadiusSubgraphs,
    SampleSubNodes]


class BaseDataModule(LightningDataModule):
    """Base LightningDataModule class.

    Child classes should overwrite:

    ```
    MyDataModule(BaseDataModule):

        _DATASET_CLASS = ...
        _MINIDATASET_CLASS = ...
    ```

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """
    _DATASET_CLASS = None
    _MINIDATASET_CLASS = None

    def __init__(
            self, data_dir='', pre_transform=None, train_transform=None,
            val_transform=None, test_transform=None,
            on_device_train_transform=None, on_device_val_transform=None,
            on_device_test_transform=None, dataloader=None, mini=False,
            trainval=False, val_on_test=False, tta_runs=None, tta_val=False,
            submit=False, **kwargs):
        super().__init__()

        # This line allows to access init params with 'self.hparams'
        # attribute also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.kwargs = kwargs

        # Make sure `_DATASET_CLASS` and `_MINIDATASET_CLASS` have been
        # specified
        if self.dataset_class is None:
            raise NotImplementedError

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Do not set the transforms directly, use self.set_transforms()
        # instead to parse the input configs
        self.pre_transform = None
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None
        self.on_device_train_transform = None
        self.on_device_val_transform = None
        self.on_device_test_transform = None

        # Instantiate the transforms
        self.set_transforms()

        # Check TTA and transforms conflicts
        self.check_tta_conflicts()

        # Check test submission and transforms conflicts
        self.check_submission_conflicts()

    @property
    def dataset_class(self):
        """Return the LightningDataModule's Dataset class.
        """
        if self.hparams.mini:
            return self._MINIDATASET_CLASS
        return self._DATASET_CLASS

    @property
    def train_stage(self):
        """Return either 'train' or 'trainval' depending on how
        `self.hparams.trainval` is configured.
        """
        return 'trainval' if self.hparams.trainval else 'train'

    @property
    def val_stage(self):
        """Return either 'val' or 'test' depending on how
        `self.hparams.val_on_test` is configured.
        """
        return 'test' if self.hparams.val_on_test else 'val'

    def prepare_data(self):
        """Download and heavy preprocessing of data should be triggered
        here.

        However, do not use it to assign state (e.g. self.x = y) because
        it will not be preserved outside this scope.
        """
        self.dataset_class(
            self.hparams.data_dir, stage=self.train_stage,
            transform=self.train_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_train_transform, **self.kwargs)

        self.dataset_class(
            self.hparams.data_dir, stage=self.val_stage,
            transform=self.val_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_val_transform, **self.kwargs)

        self.dataset_class(
            self.hparams.data_dir, stage='test',
            transform=self.test_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_test_transform, **self.kwargs)

    def setup(self, stage=None):
        """Load data. Set variables: `self.train_dataset`,
        `self.val_dataset`, `self.test_dataset`.

        This method is called by lightning with both `trainer.fit()`
        and `trainer.test()`, so be careful not to execute things like
        random split twice!
        """
        self.train_dataset = self.dataset_class(
            self.hparams.data_dir, stage=self.train_stage,
            transform=self.train_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_train_transform, **self.kwargs)

        self.val_dataset = self.dataset_class(
            self.hparams.data_dir, stage=self.val_stage,
            transform=self.val_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_val_transform, **self.kwargs)

        self.test_dataset = self.dataset_class(
            self.hparams.data_dir, stage='test',
            transform=self.test_transform, pre_transform=self.pre_transform,
            on_device_transform=self.on_device_test_transform, **self.kwargs)

        self.predict_dataset = None

    def set_transforms(self):
        """Parse in self.hparams in search for '*transform*' keys and
        instantiate the corresponding transforms.

        Credit: https://github.com/torch-points3d/torch-points3d
        """
        for key_name in self.hparams.keys():
            if "transform" in key_name:
                name = key_name.replace("transforms", "transform")
                params = getattr(self.hparams, key_name, None)
                if params is None:
                    continue
                try:
                    transform = instantiate_transforms(params)
                except Exception:
                    log.exception(f"Error trying to create {name}, {params}")
                    continue
                setattr(self, name, transform)

    def check_tta_conflicts(self):
        """Make sure the transforms are Test-Time Augmentation-friendly
        """
        # Skip if not TTA
        if self.hparams.tta_runs is None or self.hparams.tta_runs == 1:
            return

        # Make sure all transforms are test-time augmentation friendly
        transforms = getattr(self.test_transform, 'transforms', [])
        transforms += getattr(self.on_device_test_transform, 'transforms', [])
        if self.hparams.tta_val:
            transforms += getattr(self.val_transform, 'transforms', [])
            transforms += getattr(self.on_device_val_transform, 'transforms', [])
        for t in transforms:
            if t in _TTA_CONFLICTS:
                raise NotImplementedError(
                    f"Cannot use {t} with test-time augmentation. The "
                    f"following transforms are not supported: {_TTA_CONFLICTS}")

    def check_submission_conflicts(self):
        """Make sure the transforms and other parameters do not prevent
        test prediction submission.
        """
        # Skip if submission not needed
        if not self.hparams.submit:
            return

        # Make sure the dataloader only produces predictions for 1 cloud
        # at a time
        if self.hparams.dataloader.batch_size > 1:
            raise NotImplementedError(
                f"Cannot run test prediction submission for dataloaders "
                f"with batch size > 1")

        # Make sure all transforms are test submission friendly
        transforms = getattr(self.test_transform, 'transforms', [])
        transforms += getattr(self.on_device_test_transform, 'transforms', [])
        for t in transforms:
            if t in _SUBMISSION_CONFLICTS:
                raise NotImplementedError(
                    f"Cannot use {t} with test prediction submission. The "
                    f"following transforms are not supported: "
                    f"{_SUBMISSION_CONFLICTS}")

    def train_dataloader(self):
        from torch.utils.data import RandomSampler
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            persistent_workers=self.hparams.dataloader.persistent_workers,
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            persistent_workers=self.hparams.dataloader.persistent_workers,
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.dataloader.batch_size,
            num_workers=self.hparams.dataloader.num_workers,
            pin_memory=self.hparams.dataloader.pin_memory,
            persistent_workers=self.hparams.dataloader.persistent_workers,
            shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage=None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict):
        """Things to do when loading checkpoint."""
        pass

    @torch.no_grad()
    def on_after_batch_transfer(self, nag_list, dataloader_idx):
        """Intended to call on-device operations. Typically,
        NAGBatch.from_nag_list and some Transforms like SampleSubNodes
        and SampleSegments are faster on GPU, and we may prefer
        executing those on GPU rather than in CPU-based DataLoader.

        Use self.on_device_<stage>_transform, to benefit from this hook.
        """
        # Since NAGBatch.from_nag_list takes a bit of time, we asked
        # src.loader.DataLoader to simply pass a list of NAG objects,
        # waiting for to be batched on device.
        nag = NAGBatch.from_nag_list(nag_list)
        del nag_list

        # Here we run on_device_transform, which contains NAG transforms
        # that we could not / did not want to run using CPU-based
        # DataLoaders
        if self.trainer.training:
            on_device_transform = self.on_device_train_transform
        elif self.trainer.validating:
            on_device_transform = self.on_device_val_transform
        elif self.trainer.testing:
            on_device_transform = self.on_device_test_transform
        elif self.trainer.predicting:
            raise NotImplementedError('No on_device_predict_transform yet...')
        elif self.trainer.evaluating:
            on_device_transform = self.on_device_test_transform
        elif self.trainer.sanity_checking:
            on_device_transform = self.on_device_train_transform
        else:
            log.warning(
                'Unsure which stage we are in, defaulting to '
                'self.on_device_train_transform')
            on_device_transform = self.on_device_train_transform

        # Skip on_device_transform if None
        if on_device_transform is None:
            return nag

        # Apply on_device_transform only once when in training mode and
        # if no test-time augmentation is required
        if self.trainer.training \
                or self.hparams.tta_runs is None \
                or self.hparams.tta_runs == 1 or \
                (self.trainer.validating and not self.hparams.tta_val):
            return on_device_transform(nag)

        # We return the input NAG as well as the augmentation transform
        # and the number of runs. Those will be used by
        # `LightningModule.step` to accumulate multiple augmented runs
        return nag, on_device_transform, self.hparams.tta_runs

    def __repr__(self):
        return f'{self.__class__.__name__}'
