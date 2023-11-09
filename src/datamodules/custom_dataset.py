import logging
from src.datamodules.base import BaseDataModule
from src.datasets import CustomDataset, MiniCustomDataset


log = logging.getLogger(__name__)


class CustomDataModule(BaseDataModule):
    """
    Child classes should overwrite:

    ```
    MyDataModule(BaseDataModule):

        _DATASET_CLASS = ...
        _MINIDATASET_CLASS = ...
    ```
    """
    _DATASET_CLASS = CustomDataset
    _MINIDATASET_CLASS = MiniCustomDataset


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/custom_dataset.yaml")
    _ = hydra.utils.instantiate(cfg)