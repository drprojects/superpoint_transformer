import torch
from torch_scatter import scatter_add


__all__ = ['histogram_to_atomic', 'atomic_to_histogram']


def histogram_to_atomic(gt, pred):
    """Convert ground truth and predictions at a segment level (ie
    ground truth is 2D tensor carrying histogram of labels in each
    segment), to pointwise 1D ground truth and predictions.

    :param gt: 1D or 2D torch.Tensor
    :param pred: 1D or 2D torch.Tensor
    """
    assert gt.dim() <= 2

    # Edge cases where nothing happens
    if gt.dim() == 1:
        return gt, pred
    if gt.shape[1] == 1:
        return gt.squeeze(1), pred

    # Initialization
    num_nodes, num_classes = gt.shape
    device = pred.device

    # Flatten the pointwise ground truth
    point_gt = torch.arange(
        num_classes, device=device).repeat(num_nodes).repeat_interleave(
        gt.flatten())

    # Expand the pointwise ground truth
    point_pred = pred.repeat_interleave(gt.sum(dim=1), dim=0)

    return point_gt, point_pred


def atomic_to_histogram(item, idx, n_bins=None):
    """Convert point-level positive integer data to histograms of
    segment-level labels, based on idx.

    :param item: 1D or 2D torch.Tensor
    :param idx: 1D torch.Tensor
    """
    assert item.ge(0).all(), \
        "Mean aggregation only supports positive integers"
    assert item.dtype in [torch.uint8, torch.int, torch.long], \
        "Mean aggregation only supports positive integers"
    assert item.ndim <= 2, \
        "Voting and histograms are only supported for 1D and " \
        "2D tensors"

    # Initialization
    n_bins = item.max() + 1 if n_bins is None else n_bins

    # Temporarily convert input item to long
    in_dtype = item.dtype
    item = item.long()

    # Important: if values are already 2D, we consider them to
    # be histograms and will simply scatter_add them
    if item.ndim == 2:
        return scatter_add(item, idx, dim=0)

    # Convert values to one-hot encoding. Values are temporarily offset
    # to 0 to save some memory and compute in one-hot encoding and
    # scatter_add
    offset = item.min()
    item = torch.nn.functional.one_hot(item - offset)

    # Count number of occurrence of each value
    hist = scatter_add(item, idx, dim=0)
    N = hist.shape[0]
    device = hist.device

    # Prepend 0 columns to the histogram for bins removed due to
    # offsetting
    bins_before = torch.zeros(
        N, offset, device=device, dtype=torch.long)
    hist = torch.cat((bins_before, hist), dim=1)

    # Append columns to the histogram for unobserved classes/bins
    bins_after = torch.zeros(
        N, n_bins - hist.shape[1], device=device,
        dtype=torch.long)
    hist = torch.cat((hist, bins_after), dim=1)

    # Restore input dtype
    hist = hist.to(in_dtype)

    return hist
