import pyrootutils
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


__all__ = ['init_config']


def init_config(config_name='train.yaml', overrides=[]):
    # Registering the "eval" resolver allows for advanced config
    # interpolation with arithmetic operations:
    # https://omegaconf.readthedocs.io/en/2.3_branch/how_to_guides.html
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval', eval)

    GlobalHydra.instance().clear()
    pyrootutils.setup_root(".", pythonpath=True)
    with initialize(version_base='1.2', config_path="../../configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
