import logging
from src.datamodules.base import BaseDataModule
from src.datasets import UKLidarDataset


log = logging.getLogger(__name__)


class UKLidarDataModule(BaseDataModule):
    """LightningDataModule for UK LiDAR dataset."""
    _DATASET_CLASS = UKLidarDataset
    _MINIDATASET_CLASS = None
