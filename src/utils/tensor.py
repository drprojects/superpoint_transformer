import torch
import numpy as np
from numba import njit


__all__ = [
    'tensor_idx', 'is_sorted', 'has_duplicates', 'is_dense', 'is_permutation',
    'arange_interleave', 'print_tensor_info', 'cast_to_optimal_integer_type',
    'cast_numpyfy', 'numpyfy', 'torchify', 'torch_to_numpy', 'fast_randperm',
    'fast_zeros', 'fast_repeat', 'string_to_dtype']


def tensor_idx(idx, device=None):
    """Convert an int, slice, list or numpy index to a torch.LongTensor.
    """
    if device is None and hasattr(idx, 'device'):
        device = idx.device
    elif device is None:
        device = 'cpu'

    if idx is None:
        idx = torch.tensor([], device=device, dtype=torch.long)
    elif isinstance(idx, int):
        idx = torch.tensor([idx], device=device, dtype=torch.long)
    elif isinstance(idx, list):
        idx = torch.tensor(idx, device=device, dtype=torch.long)
    elif isinstance(idx, slice):
        idx = torch.arange(idx.stop, device=device)[idx]
    elif isinstance(idx, np.ndarray):
        idx = torch.from_numpy(idx).to(device)
    # elif not isinstance(idx, torch.LongTensor):
    #     raise NotImplementedError

    if isinstance(idx, torch.BoolTensor):
        idx = torch.where(idx)[0]

    assert idx.dtype is torch.int64, \
        f"Expected LongTensor but got {idx.dtype} instead."
    # assert idx.shape[0] > 0, \
    #     "Expected non-empty indices. At least one index must be provided."

    return idx


def is_sorted(a: torch.LongTensor, increasing=True, strict=False):
    """Checks whether a 1D tensor of indices is sorted."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    if increasing and strict:
        f = torch.gt
    if increasing and not strict:
        f = torch.ge
    if not increasing and strict:
        f = torch.lt
    if not increasing and not strict:
        f = torch.le
    return f(a[1:], a[:-1]).all()


def has_duplicates(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices contains duplicates."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    return a.unique().numel() != a.numel()


def is_dense(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices contains dense indices.
    That is to say all values in [a.min(), a.max] appear at least once
    in a.
    """
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    unique = a.unique()
    return a.min() == 0 and unique.size(0) == a.max().long() + 1


def is_permutation(a: torch.LongTensor):
    """Checks whether a 1D tensor of indices is a permutation."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    return a.sort().values.long().equal(torch.arange(a.numel(), device=a.device))


def arange_interleave(width, start=None):
    """Vectorized equivalent of:
        >>> torch.cat([torch.arange(s, s + w) for w, s in zip(width, start)])
    """
    assert width.dim() == 1, 'Only supports 1D tensors'
    assert isinstance(width, torch.Tensor), 'Only supports Tensors'
    assert not width.is_floating_point(), 'Only supports Tensors of integers'
    assert width.ge(0).all(), 'Only supports positive integers'
    start = start if start is not None else torch.zeros_like(width)
    assert width.shape == start.shape
    assert start.dim() == 1, 'Only supports 1D tensors'
    assert isinstance(start, torch.Tensor), 'Only supports Tensors'
    assert not start.is_floating_point(), 'Only supports Tensors of integers'
    width = width.long()
    start = start.long()
    device = width.device
    a = torch.cat((torch.zeros(1, device=device).long(), width[:-1]))
    offsets = (start - a.cumsum(0)).repeat_interleave(width)
    return torch.arange(width.sum(), device=device) + offsets


def print_tensor_info(a, name=None):
    """Print some info about a tensor. Used for debugging.
    """
    is_1d = a.dim() == 1
    is_int = not a.is_floating_point()

    msg = f'{name}:  ' if name is not None else ''

    msg += f'shape={a.shape}  '
    msg += f'dtype={a.dtype}  '
    msg += f'min={a.min()}  '
    msg += f'max={a.max()}  '

    if is_1d and is_int:
        msg += f'duplicates={has_duplicates(a)}  '
        msg += f'sorted={is_sorted(a)}  '
        msg += f'dense={is_dense(a)}  '
        msg += f'permutation={is_permutation(a)}  '

    print(msg)


def string_to_dtype(string):
    if isinstance(string, torch.dtype):
        return string
    assert isinstance(string, str)
    if string in ('half', 'float16'):
        return torch.float16
    if string in ('float', 'float32'):
        return torch.float32
    if string in ('double', 'float64'):
        return torch.float64
    if string == 'bool':
        return torch.bool
    if string in ('byte', 'uint8'):
        return torch.uint8
    if string in ('byte', 'int8'):
        return torch.int8
    if string in ('short', 'int16'):
        return torch.float16
    if string in ('int', 'int32'):
        return torch.float32
    if string in ('long', 'int64'):
        return torch.float64
    raise ValueError(f"Unknown dtype='{string}'")


def cast_to_optimal_integer_type(a):
    """Cast an integer tensor to the smallest possible integer dtype
    preserving its precision.
    """
    assert isinstance(a, torch.Tensor), \
        f"Expected an Tensor input, but received {type(a)} instead"
    assert not a.is_floating_point(), \
        f"Expected an integer-like input, but received dtype={a.dtype} instead"

    for dtype in [torch.uint8, torch.int16, torch.int32, torch.int64]:
        low_enough = torch.iinfo(dtype).min <= a.min()
        high_enough = a.max() <= torch.iinfo(dtype).max
        if low_enough and high_enough:
            return a.to(dtype)

    raise ValueError(f"Could not cast dtype={a.dtype} to integer.")


def cast_numpyfy(a, fp_dtype=torch.float):
    """Convert torch.Tensor to numpy while respecting some constraints
    on output dtype. Integer tensors will be cast to the smallest
    possible integer dtype preserving their precision. Floating point
    tensors will be cast to `fp_dtype`.
    """
    if not isinstance(a, torch.Tensor):
        return numpyfy(a)

    # Convert string dtype to torch dtype, if need be
    fp_dtype = string_to_dtype(fp_dtype)

    # Rule out non-floating-point tensors
    if not a.is_floating_point():
        return numpyfy(cast_to_optimal_integer_type(a))

    # Cast floating point tensors
    return numpyfy(a.to(fp_dtype))


def numpyfy(a):
    """Convert torch.Tensor to numpy while respecting some constraints
    on output dtype.
    """
    if not isinstance(a, torch.Tensor):
        return a

    return a.cpu().numpy()


def torchify(x):
    """Convert np.ndarray to torch.Tensor.
    """
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


def torch_to_numpy(func):
    """Decorator intended for numpy-based functions to be fed and return
    torch.Tensor arguments.

    :param func:
    :return:
    """

    def wrapper_torch_to_numba(*args, **kwargs):
        args_numba = [numpyfy(x) for x in args]
        kwargs_numba = {k: numpyfy(v) for k, v in kwargs.items()}
        out = func(*args_numba, **kwargs_numba)
        if isinstance(out, list):
            out = [torchify(x) for x in out]
        elif isinstance(out, tuple):
            out = tuple([torchify(x) for x in list(out)])
        elif isinstance(out, dict):
            out = {k: torchify(v) for k, v in out.items()}
        else:
            out = torchify(out)
        return out

    return wrapper_torch_to_numba


@torch_to_numpy
@njit(cache=True, nogil=True)
def numba_randperm(n):
    """Same as torch.randperm but leveraging numba on CPU.

    NB: slightly faster than `np.random.permutation(np.arange(n))`
    """
    a = np.arange(n)
    np.random.shuffle(a)
    return a


def fast_randperm(n, device='cpu'):
    """Same as torch.randperm, but relies on numba for CPU tensors. This
    may bring a x2 speedup on CPU for n >= 1e5.

    ```
    from time import time
    import torch
    from src.utils.tensor import fast_randperm

    n = 100000

    start = time()
    a = torch.randperm(n)
    print(f'torch.randperm : {time() - start:0.5f}s')

    start = time()
    b = fast_randperm(n)
    print(f'fast_randperm: {time() - start:0.5f}s')
    ```
    """
    if device == 'cuda' or \
            isinstance(device, torch.device) and device.type == 'cuda':
        return torch.randperm(n, device=device)
    return numba_randperm(n)


# Not working as good as experiments promised...
def fast_zeros(*args, dtype=None, device='cpu'):
    """Same as torch.zeros but relies numpy on CPU. This may be x40
    faster when manipulating large tensors on CPU.

    ```
    from time import time
    import torch
    import numpy as np
    from src.utils.tensor import fast_zeros

    n = 1000000
    m = 20

    start = time()
    a = torch.zeros(n, m)
    print(f'torch.zeros : {time() - start:0.4f}s')

    start = time()
    b = torch.from_numpy(np.zeros((n, m), dtype='float32'))
    print(f'np.zeros: {time() - start:0.4f}s')

    start = time()
    c = fast_zeros(n, m)
    print(f'fast_zeros: {time() - start:0.4f}s')

    print(torch.equal(a, b), torch.equal(a, c))
    ```
    """
    if device == 'cuda' or \
        isinstance(device, torch.device) and device.type == 'cuda':
        return torch.zeros(*args, dtype=dtype, device=device)
    out = torchify(np.zeros(tuple(args), dtype='float32'))
    if dtype is not None:
        out = out.to(dtype)
    return out


def fast_repeat(x, repeats):
    """Same as torch.repeat_interleave but relies numpy on CPU. This
    saves a little bit of time when manipulating large tensors on CPU.

    ```
    from time import time
    import torch
    import numpy as np
    from src.utils.tensor import fast_repeat

    n = 1000000
    rmax = 50
    values = torch.arange(n)
    repeats = torch.randint(low=0, high=rmax, size=(n,))

    start = time()
    a = values.repeat_interleave(repeats)
    print(f'torch.repeat_interleave : {time() - start:0.4f}s')

    start = time()
    b = torch.from_numpy(np.repeat(values.numpy(), repeats.numpy()))
    print(f'np.repeat: {time() - start:0.4f}s')

    start = time()
    c = fast_repeat(values, repeats)
    print(f'fast_repeat: {time() - start:0.4f}s')

    print(torch.equal(a, b), torch.equal(a, c))
    ```
    """
    assert isinstance(x, torch.Tensor)
    assert isinstance(repeats, int) or x.device == repeats.device
    if x.is_cuda:
        return torch.repeat_interleave(x, repeats)
    if isinstance(repeats, int):
        return torchify(np.repeat(numpyfy(x), repeats))
    else:
        return torchify(np.repeat(numpyfy(x), numpyfy(repeats)))
