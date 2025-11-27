import sys
import hydra
import torch
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from src.utils.hydra import init_config

from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils.point import is_xyz_tensor
from omegaconf import OmegaConf


__all__ = ['xy_partition', 'xyz_partition', 'compute_partition_purity_metrics_s3dis_6fold']


def xy_partition(pos, grid, consecutive=True):
    """Partition a point cloud based on a regular XY grid. Returns, for
    each point, the index of the grid cell it falls into.

    :param pos: Tensor
        Point cloud
    :param grid: float
        Grid size
    :param consecutive: bool
        Whether the grid cell indices should be consecutive. That is to
        say all indices in [0, idx_max] are used. Note that this may
        prevent trivially mapping an index value back to the
        corresponding XY coordinates
    :return:
    """
    assert is_xyz_tensor(pos)

    # Compute the (i, j) coordinates on the XY grid size
    i = pos[:, 0].div(grid, rounding_mode='trunc').long()
    j = pos[:, 1].div(grid, rounding_mode='trunc').long()

    # Shift coordinates to positive integer to avoid negatives
    # clashing with our downstream indexing mechanism
    i -= i.min()
    j -= j.min()

    # Compute a "manual" partition based on the grid coordinates
    super_index = i * (max(i.max(), j.max()) + 1) + j

    # If required, update the used indices to be consecutive
    if consecutive:
        super_index = consecutive_cluster(super_index)[0]

    return super_index


def xyz_partition(pos, grid, consecutive=True):
    """Partition a point cloud based on a regular XYZ grid. Returns, for
    each point, the index of the grid cell it falls into.
    """
    assert is_xyz_tensor(pos)
    
    # Compute the (i, j, k) coordinates on the XYZ grid size
    i = pos[:, 0].div(grid, rounding_mode='trunc').long()
    j = pos[:, 1].div(grid, rounding_mode='trunc').long()
    k = pos[:, 2].div(grid, rounding_mode='trunc').long()
    
    # Shift coordinates to positive integer to avoid negatives
    # clashing with our downstream indexing mechanism
    i -= i.min()
    j -= j.min()
    k -= k.min()
    
    # Compute a "manual" partition based on the grid coordinates
    super_index = i * (max(i.max(), j.max()) + 1) * (max(i.max(), j.max(), k.max()) + 1) + j * (max(i.max(), j.max(), k.max()) + 1) + k
    
    # If required, update the used indices to be consecutive
    if consecutive:
        super_index = consecutive_cluster(super_index)[0]

    return super_index

def compute_partition_purity_metrics_s3dis_6fold(
        config,
        level,
        verbose=False):
    
    """Helper function to compute the partition purity metrics of a
    model on a S3DIS 6-fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Local import to avoid import loop errors
    from src.metrics import ConfusionMatrix
    from torchmetrics import SumMetric
    from src.data import NAGBatch

    # Very ugly fix to ignore lightning's warning messages about the
    # trainer and modules not being connected
    import warnings
    warnings.filterwarnings("ignore")
    
    n_p_metric = SumMetric().to(device)
    n_sp_metric = SumMetric().to(device)
    cm_list = []
    
    num_classes = None
    
    for fold in range(1, 7):
        if verbose:
            print(f"\nFold {fold}")

        # Parse the configs using hydra
        cfg = init_config(overrides=[
            f"experiment={config}",
            f"datamodule.fold={fold}",])
        
        # print(OmegaConf.to_yaml(cfg))
        
        # Instantiate the datamodule
        datamodule = hydra.utils.instantiate(cfg.datamodule)

        # Compute the preprocessing only for the val
        datamodule.val_dataset = datamodule.dataset_class(
            datamodule.hparams.data_dir, stage=datamodule.val_stage,
            transform=datamodule.val_transform, pre_transform=datamodule.pre_transform,
            on_device_transform=datamodule.on_device_val_transform, **datamodule.kwargs)
        val_dataset = datamodule.val_dataset
        
        print(f"val_stage : {osp.join(val_dataset.processed_dir, datamodule.val_stage, val_dataset.pre_transform_hash)}")


        # Gather some details from the model and datamodule before
        # deleting them
        num_classes = val_dataset.num_classes
        
        fold_cm = ConfusionMatrix(num_classes).to(device)
        
        dataloader = datamodule.val_dataloader()
        enum = tqdm(dataloader) if verbose else dataloader
        for nag_list in enum:
            
            nag = NAGBatch.from_nag_list([nag.to(device) for nag in nag_list])
            nag = val_dataset.on_device_transform(nag)
            
            y_oracle = nag[level].y[:, :num_classes].argmax(dim=1)
            fold_cm.update(pred=y_oracle, target=nag[level].y)
            n_p_metric.update(nag[0].num_nodes)
            n_sp_metric.update(nag[level].num_nodes)

        # Store the metrics for each fold
        cm_list.append(fold_cm)
        print(f"fold {fold} omiou: {fold_cm.miou().cpu().item():.1f}")
        print(f"with {nag[level].num_nodes:.0f} superpoints (ratio={nag[0].num_nodes/nag[level].num_nodes:.0f})")

        del datamodule, val_dataset
    
    # Initialize the 6-fold metrics
    semantic_6fold = ConfusionMatrix(num_classes)

    # Group together per-fold panoptic and semantic results
    for i in range(len(cm_list)):
        semantic_6fold.confmat += cm_list[i].confmat.cpu()

    # Print computed the metrics
    print(f"\n6-fold")
    print(f"OmIoU : {semantic_6fold.miou().cpu().item():.1f}")
    print(f"OOA   : {semantic_6fold.oa().cpu().item():.1f}")
    print(f"OmAcc : {semantic_6fold.macc().cpu().item():.1f}")
    print(f"n_sp : {n_sp_metric.compute().cpu().item():.0f}")
    print(f"n_p : {n_p_metric.compute().cpu().item():.0f}")
    print(f"ratio : {n_p_metric.compute().cpu().item()/n_sp_metric.compute().cpu().item():.0f}")
    
    return semantic_6fold, cm_list