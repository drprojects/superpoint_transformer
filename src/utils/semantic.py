import hydra
import torch
from tqdm import tqdm
from copy import deepcopy
from src.utils.hydra import init_config


__all__ = ['compute_semantic_metrics', 'compute_semantic_metrics_s3dis_6fold']


def compute_semantic_metrics(
        model,
        datamodule,
        stage='val',
        verbose=True):
    """Helper function to compute the semantic segmentation metrics of a
    model on a given dataset.
    """
    # Local imports to avoid import loop errors
    from src.data import NAGBatch

    # Pick among train, val, and test datasets. It is important to note
    # that the train dataset produces augmented spherical samples of
    # large scenes, while the val and test dataset
    if stage == 'train':
        dataset = datamodule.train_dataset
        dataloader = datamodule.train_dataloader()
    elif stage == 'val':
        dataset = datamodule.val_dataset
        dataloader = datamodule.val_dataloader()
    elif stage == 'test':
        dataset = datamodule.test_dataset
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unknown stage : {stage}")

    # Prevent `NAGAddKeysTo` from removing attributes to allow
    # visualizing them after model inference
    dataset = _set_attribute_preserving_transforms(dataset)

    # Load a dataset item. This will return the hierarchical partition
    # of an entire tile, within a NAG object
    with torch.no_grad():
        enum = tqdm(dataloader) if verbose else dataloader
        for nag_list in enum:
            nag = NAGBatch.from_nag_list([nag.cuda() for nag in nag_list])

            # Apply on-device transforms on the NAG object. For the
            # train dataset, this will select a spherical sample of the
            # larger tile and apply some data augmentations. For the
            # validation and test datasets, this will prepare an entire
            # tile for inference
            nag = dataset.on_device_transform(nag)

            # NB: we use the "validation_step" protocol here, regardless
            # of the stage the data comes from
            model.validation_step(nag, None)

        # Actions taken from on_validation_epoch_end()
        semantic = deepcopy(model.val_cm)
        model.val_cm.reset()

    if not verbose:
        return semantic

    print(f"mIoU : {semantic.miou().cpu().item()}")
    print(f"OA   : {semantic.oa().cpu().item()}")
    print(f"mAcc : {semantic.macc().cpu().item()}")

    return semantic


def compute_semantic_metrics_s3dis_6fold(
        fold_ckpt,
        experiment_config,
        stage='val',
        verbose=False):
    """Helper function to compute the semantic segmentation metrics of a
    model on a S3DIS 6-fold.

    :param fold_ckpt: dict
        Dictionary with S3DIS fold numbers as keys and checkpoint paths
        as values
    :param experiment_config: str
        Experiment config to use for inference. For instance for S3DIS
        with semantic segmentation: 'semantic/s3dis'
    :param stage: str
    :param verbose: bool
    :return:
    """
    # Local import to avoid import loop errors
    from src.metrics import ConfusionMatrix

    # Very ugly fix to ignore lightning's warning messages about the
    # trainer and modules not being connected
    import warnings
    warnings.filterwarnings("ignore")

    semantic_list = []
    num_classes = None

    for fold, ckpt_path in fold_ckpt.items():

        if verbose:
            print(f"\nFold {fold}")

        # Parse the configs using hydra
        cfg = init_config(overrides=[
            f"experiment={experiment_config}",
            f"datamodule.fold={fold}",
            f"ckpt_path={ckpt_path}"])

        # Instantiate the datamodule
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.prepare_data()
        datamodule.setup()

        # Instantiate the model
        model = hydra.utils.instantiate(cfg.model)

        # Load pretrained weights from a checkpoint file
        model = model._load_from_checkpoint(cfg.ckpt_path)
        model = model.eval().cuda()

        # Compute metrics on the fold
        semantic = compute_semantic_metrics(
            model,
            datamodule,
            stage=stage,
            verbose=verbose)

        # Gather some details from the model and datamodule before
        # deleting them
        num_classes = datamodule.train_dataset.num_classes

        del model, datamodule

        # Store the metrics for each fold
        semantic_list.append(semantic)

    # Initialize the 6-fold metrics
    semantic_6fold = ConfusionMatrix(num_classes)

    # Group together per-fold panoptic and semantic results
    for i in range(len(semantic_list)):
        semantic_6fold.confmat += semantic_list[i].confmat.cpu()
        
    # Display the metrics again
    for i, cm in enumerate(semantic_list):
        print(f"fold {i+1} mIoU: {cm.miou().cpu().item():.1f}")

    # Print computed the metrics
    print(f"\n6-fold")
    print(f"mIoU : {semantic_6fold.miou().cpu().item():.1f}")
    print(f"OA   : {semantic_6fold.oa().cpu().item():.1f}")
    print(f"mAcc : {semantic_6fold.macc().cpu().item():.1f}")

    return semantic_6fold, semantic_list


def _set_attribute_preserving_transforms(dataset):
    """For the sake of visualization, we require that `NAGAddKeysTo`
    does not remove input `Data` attributes after moving them to
    `Data.x`, so we may visualize them.
    """
    # Local imports to avoid import loop errors
    from src.transforms import NAGAddKeysTo

    for t in dataset.on_device_transform.transforms:
        if isinstance(t, NAGAddKeysTo):
            t.delete_after = False

    return dataset
