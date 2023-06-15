import logging
from src.datamodules.base import BaseDataModule
from src.datasets import S3DISRoom, MiniS3DISRoom


# Occasional Dataloader issues with S3DISRoomDataModule on some
# machines. Hack to solve this:
# https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


log = logging.getLogger(__name__)


class S3DISRoomDataModule(BaseDataModule):
    """LightningDataModule for S3DIS dataset.

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
    _DATASET_CLASS = S3DISRoom
    _MINIDATASET_CLASS = MiniS3DISRoom


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/s3dis_room.yaml")
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)
