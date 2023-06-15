import torch
import src
from src.utils.tensor import is_dense, is_sorted, fast_repeat, tensor_idx, \
    arange_interleave, fast_randperm
from torch_scatter import scatter_mean


__all__ = [
    'indices_to_pointers', 'sizes_to_pointers', 'dense_to_csr', 'csr_to_dense',
    'sparse_sort', 'sparse_sort_along_direction', 'sparse_sample']


def indices_to_pointers(indices: torch.LongTensor):
    """Convert pre-sorted dense indices to CSR format."""
    device = indices.device
    assert len(indices.shape) == 1, "Only 1D indices are accepted."
    assert indices.shape[0] >= 1, "At least one group index is required."
    assert is_dense(indices), "Indices must be dense"

    # Sort indices if need be
    order = torch.arange(indices.shape[0], device=device)
    if not is_sorted(indices):
        indices, order = indices.sort()

    # Convert sorted indices to pointers
    pointers = torch.cat([
        torch.LongTensor([0]).to(device),
        torch.where(indices[1:] > indices[:-1])[0] + 1,
        torch.LongTensor([indices.shape[0]]).to(device)])

    return pointers, order


def sizes_to_pointers(sizes: torch.LongTensor):
    """Convert a tensor of sizes into the corresponding pointers. This
    is a trivial but often-required operation.
    """
    assert sizes.dim() == 1
    assert sizes.dtype == torch.long
    zero = torch.zeros(1, device=sizes.device, dtype=torch.long)
    return torch.cat((zero, sizes)).cumsum(dim=0)


def dense_to_csr(a):
    """Convert a dense matrix to its CSR counterpart."""
    assert a.dim() == 2
    index = a.nonzero(as_tuple=True)
    values = a[index]
    columns = index[1]
    pointers = indices_to_pointers(index[0])[0]
    return pointers, columns, values


def csr_to_dense(pointers, columns, values, shape=None):
    """Convert a CSR matrix to its dense counterpart of a given shape.
    """
    assert pointers.dim() == 1
    assert columns.dim() == 1
    assert values.dim() == 1
    assert shape is None or len(shape) == 2
    assert pointers.device == columns.device == values.device

    device = pointers.device

    shape_guess = (pointers.shape[0] - 1, columns.max() + 1)
    if shape is None:
        shape = shape_guess
    else:
        shape = (max(shape[0], shape_guess[0]), max(shape[1], shape_guess[1]))

    n, m = shape
    a = torch.zeros(n, m, dtype=values.dtype, device=device)
    i = torch.arange(n, device=device)
    i = fast_repeat(i, pointers[1:] - pointers[:-1])
    j = columns.long()
    a[i, j] = values

    return a


def sparse_sort(src, index, dim=0, descending=False, eps=1e-6):
    """Lexicographic sort 1D src points based on index first and src
    values second.

    Credit: https://github.com/rusty1s/pytorch_scatter/issues/48
    """
    # NB: we use double precision here to make sure we can capture fine
    # grained src changes even with very large index values.
    f_src = src.double()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min)/(f_max - f_min + eps) + index.double()*(-1)**int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm


def sparse_sort_along_direction(src, index, direction, descending=False):
    """Lexicographic sort N-dimensional src points based on index first
    and the projection of the src values along a direction second.
    """
    assert src.dim() == 2
    assert index.dim() == 1
    assert src.shape[0] == index.shape[0]
    assert direction.dim() == 2 or direction.dim() == 1

    if direction.dim() == 1:
        direction = direction.view(1, -1)

    # If only 1 direction is provided, apply the same direction to all
    # points
    if direction.shape[0] == 1:
        direction = direction.repeat(src.shape[0], 1)

    # If the direction is provided group-wise, expand it to the points
    if direction.shape[0] != src.shape[0]:
        direction = direction[index]

    # Compute the centroid for each group. This is not mandatory, but
    # may help avoid precision errors if absolute src coordinates are
    # too large
    centroid = scatter_mean(src, index, dim=0)[index]

    # Project the points along the associated direction
    projection = torch.einsum('ed, ed -> e', src - centroid, direction)

    # Sort the projections
    _, perm = sparse_sort(projection, index, descending=descending)

    return src[perm], perm


def sparse_sample(idx, n_max=32, n_min=1, mask=None, return_pointers=False):
    """Compute indices to sample elements in a set of size `idx.shape`,
    based on which segment they belong to in `idx`.

    The sampling operation is run without replacement and each
    segment is sampled at least `n_min` and at most `n_max` times,
    within the limits allowed by its actual size.

    Optionally, a `mask` can be passed to filter out some elements.

    :param idx: LongTensor of size N
        Segment indices for each of the N elements
    :param n_max: int
        Maximum number of elements to sample in each segment
    :param n_min: int
        Minimum number of elements to sample in each segment, within the
        limits of its size (ie no oversampling)
    :param mask: list, np.ndarray, torch.Tensor
        Indicates a subset of elements to consider. This allows ignoring
        some segments
    :param return_pointers: bool
        Whether pointers should be returned along with sampling
        indices. These indicate which sampled element belongs to which
        segment
    """
    assert 0 <= n_min <= n_max

    # Initialization
    device = idx.device
    size = idx.bincount()
    num_elements = size.sum()
    num_segments = idx.max() + 1

    # Compute the number of elements that will be sampled from each
    # segment, based on a heuristic
    if n_max > 0:
        # k * tanh(x / k) is bounded by k, is ~x for x~0 and starts
        # saturating at x~k
        n_samples = (n_max * torch.tanh(size / n_max)).floor().long()
    else:
        # Fallback to sqrt sampling
        n_samples = size.sqrt().round().long()

    # Make sure each segment is sampled at least 'n_min' times and not
    # sampled more than its size (we sample without replacements).
    # If a segment has less than 'n_min' elements, it will be
    # entirely sampled (no randomness for sampling this segment),
    # which is why we successively apply clamp min and clamp max
    n_samples = n_samples.clamp(min=n_min).clamp(max=size)

    # Sanity check
    if src.is_debug_enabled():
        assert n_samples.le(size).all(), \
            "Cannot sample more than the segment sizes."

    # Prepare the sampled elements indices
    sample_idx = torch.arange(num_elements, device=device)

    # If a mask is provided, only keep the corresponding elements.
    # This also requires updating the `size` and `n_samples`
    mask = tensor_idx(mask, device=device)
    if mask.shape[0] > 0:
        sample_idx = sample_idx[mask]
        idx = idx[mask]
        size = idx.bincount(minlength=num_segments)
        n_samples = n_samples.clamp(max=size)

    # Sanity check
    if src.is_debug_enabled():
        assert n_samples.le(size).all(), \
            "Cannot sample more than the segment sizes."

    # Shuffle the order of elements to introduce randomness
    perm = fast_randperm(sample_idx.shape[0], device=device)
    idx = idx[perm]
    sample_idx = sample_idx[perm]

    # Sort by idx. Combined with the previous shuffling,
    # this ensures the randomness in the elements selected from each
    # segment
    idx, order = idx.sort()
    sample_idx = sample_idx[order]

    # Build the indices of the elements we will sample from
    # sample_idx. Note this could easily be expressed with a for
    # loop, but we need to use a vectorized formulation to ensure
    # reasonable processing time
    offset = sizes_to_pointers(size[:-1])
    idx_samples = sample_idx[arange_interleave(n_samples, start=offset)]

    # Return here if sampling pointers are not required
    if not return_pointers:
        return idx_samples

    # Compute the pointers
    ptr_samples = sizes_to_pointers(n_samples)

    return idx_samples, ptr_samples.contiguous()
