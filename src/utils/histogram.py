import math
import torch
from torch_scatter import scatter_sum
from torch_geometric.nn.pool.consecutive import consecutive_cluster


__all__ = [
    'histogram_to_atomic',
    'atomic_to_histogram',
    'split_histogram']


def histogram_to_atomic(gt, pred):
    """Convert ground truth and predictions at a segment level (i.e.
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
    N = idx.max() + 1
    n_bins = item.max() + 1 if n_bins is None else n_bins
    device = item.device

    # Cast item to long values to avoid overflow, since we will be
    # summing values
    # TODO: a more memory efficient implementation could cast to the
    #  optimal integer dtypes. But we leave this for later
    item = item.long()

    # Important: if values are already 2D, we consider them to
    # be histograms and will simply scatter_sum them
    if item.ndim == 2:
        return scatter_sum(item, idx, dim=0)

    # Aggregate the same-item same-idx values in a 2D histogram
    hist_idx = n_bins * idx + item
    hist_idx_2, perm = consecutive_cluster(hist_idx)
    counts = hist_idx_2.bincount()
    hist = torch.zeros((N, n_bins), dtype=torch.long, device=device)
    i = hist_idx[perm] // n_bins
    j = hist_idx[perm] % n_bins
    hist[i, j] = counts

    # Append columns to the histogram for unobserved classes/bins
    bins_after = torch.zeros(
        N, n_bins - hist.shape[1],
        device=device,
        dtype=torch.long)
    hist = torch.cat((hist, bins_after), dim=1)

    return hist


def split_histogram(hist: torch.Tensor, chunk_size: int):
    """Search for the indices to partition the bins of a histogram such
    that the resulting partition splits the histogram into groups of
    chunk_size elements.

    This can typically be useful for partitioning data in smaller chunks
    based on an indexing tensor of integer.

    Importantly, this algorithm does NOT guarantee that the produced
    splits will have chunk_size at most. Actually, it can produce splits
    of up to `chunk_size + max_bin_size` in the worst case scenario.

    :param hist: torch.Tensor
    :param chunk_size: int
    """
    device = hist.device
    N = hist.sum()

    # Compute the values at which we would ideally like to split the
    # cumulative histogram
    num_chunks = math.ceil(N / chunk_size)
    v = torch.arange(0, num_chunks + 1, device=device) * chunk_size

    # Find the indices at which we can cut the cumulative histogram
    # NB: this does NOT guarantee that the produced splits will have
    # chunk_size at most. Actually, it can produce splits of up to
    # `chunk_size + max_bin_size` in the worst case scenario
    idx = torch.searchsorted(hist.cumsum(dim=0), v, right=True)
    # idx = idx[torch.where(idx[:-1] < idx[1:])]
    # idx = idx[idx > 0]

    # Compute the split indices, this is a list of tensor views
    # Clean up by removing potentially empty splits
    full_idx = torch.arange(hist.numel(), device=device)
    splits = [
        full_idx[start:end] for start, end in zip(
            torch.cat([torch.tensor([0], device=device), idx]),
            torch.cat([idx, torch.tensor([hist.numel()], device=device)])
        ) if end - start > 0]

    return splits
