import pyrootutils

root = str(
    pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "README.md"],
        pythonpath=True,
        dotenv=True,
    )
)

# Hack importing pandas here to bypass some conflicts with hydra
import pandas as pd

from typing import List

import hydra
import torch
import torch_geometric
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

from src import utils

# Registering the "eval" resolver allows for advanced config
# interpolation with arithmetic operations:
# https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
OmegaConf.register_new_resolver("eval", eval)

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    if float(".".join(torch.__version__.split(".")[:2])) >= 2.0:
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch_geometric.compile(model)  # type: ignore

    log.info("Starting predicting!")

    predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
        )  # type: ignore

    return None


@hydra.main(version_base="1.2", config_path=root + "/configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    predict(cfg)


if __name__ == "__main__":
    main()
