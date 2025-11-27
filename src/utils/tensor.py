import torch
import numpy as np
from numba import njit

from src.utils.dict import next_incremental_key, check_incremental_keys


__all__ = [
    'tensor_idx',
    'is_sorted',
    'has_duplicates',
    'is_dense',
    'is_permutation',
    'arange_interleave',
    'print_tensor_info',
    'cast_to_optimal_integer_type',
    'cast_numpyfy',
    'numpyfy',
    'torchify',
    'torch_to_numpy',
    'fast_randperm',
    'fast_zeros',
    'fast_repeat',
    'string_to_dtype',
    'float64_to_float32_pair',
    'float32_pair_to_float64',
    'to_flat_tensor',
    'from_flat_tensor',
    'cast_tensor',
    'is_arange']


def tensor_idx(idx, device=None):
    """Convert an int, slice, list or numpy index to a torch.LongTensor.
    """
    if not isinstance(idx, (int, slice, np.ndarray, torch.Tensor, type(None))):
        raise ValueError(
            f"Expected an int, slice, list, np.ndarray, or torch.Tensor "
            f"index, but received a {type(idx)} instead.")
    if device is None and hasattr(idx, 'device'):
        device = idx.device
    elif device is None:
        device = 'cpu'

    if idx is None:
        return None

    if isinstance(idx, int):
        idx = torch.tensor([idx], device=device, dtype=torch.long)
    elif isinstance(idx, list):
        idx = torch.tensor(idx, device=device, dtype=torch.long)
    elif isinstance(idx, slice):
        idx = torch.arange(idx.start, idx.stop, device=device)
    elif isinstance(idx, np.ndarray):
        idx = torch.from_numpy(idx).to(device)
    # elif not isinstance(idx, torch.LongTensor):
    #     raise NotImplementedError

    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]

    # Make sure the indices are held in a LongTensor
    idx = idx.long()

    # assert idx.dtype is torch.int64, \
    #     f"Expected LongTensor but got {idx.dtype} instead."
    # assert idx.shape[0] > 0, \
    #     "Expected non-empty indices. At least one index must be provided."

    return idx


def is_arange(a: torch.Tensor, n: int):
    """Checks whether a tensor is equal to `torch.arange(0, n)`.

    This may can come in handy when checking whether an indexing tensor
    actually modifies the object it is operating on.
    """
    return a.equal(torch.arange(n, device=a.device))


def is_sorted(a: torch.Tensor, increasing=True, strict=False):
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


def has_duplicates(a: torch.Tensor):
    """Checks whether a 1D tensor of indices contains duplicates."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    return a.unique().numel() != a.numel()


def is_dense(a: torch.Tensor):
    """Checks whether a 1D tensor of indices contains dense indices.
    That is to say all values in [0, a.max] appear at least once in a.
    """
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    assert a.numel() > 0, "0-dimensional tensors are not supported"
    unique = a.unique()
    return a.min() == 0 and unique.size(0) == a.max().long() + 1


def is_permutation(a: torch.Tensor):
    """Checks whether a 1D tensor of indices is a permutation."""
    assert a.dim() == 1, "Only supports 1D tensors"
    assert not a.is_floating_point(), "Float tensors are not supported"
    return is_arange(a.sort().values.long(), a.numel())


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


def float64_to_float32_pair(tensor):
    """Split a float64 tensor into two float32 tensors (high and low
    components) such that the original tensor can be reconstructed
    without loss of precision.

    :param tensor: torch.Tensor
        Input tensor of dtype float64
    :return: (torch.Tensor, torch.Tensor)
        Two float32 tensors representing the high and low parts
    """
    # Ensure the input tensor is float64
    assert tensor.dtype == torch.float64, "Input tensor must be of type float64"

    # Split the double-precision values into high and low components
    high = tensor.float()  # Convert to float32 (high part)
    low = tensor - high.double()  # Compute the residual (low part)

    return high, low.float()


def float32_pair_to_float64(high, low):
    """Reconstruct the original float64 tensor from two float32 tensors.

    :param high: torch.Tensor
        High part (float32)
    :param low: torch.Tensor
        Low part (float32)
    :return: torch.Tensor
        Reconstructed tensor of dtype float64
    """
    return high.double() + low.double()


def cast_to_optimal_integer_type(a):
    """Cast an integer tensor to the smallest possible integer dtype
    preserving its precision.
    """
    assert isinstance(a, torch.Tensor), \
        f"Expected an Tensor input, but received {type(a)} instead"
    assert not a.is_floating_point(), \
        f"Expected an integer-like input, but received dtype={a.dtype} instead"

    if a.numel() == 0:
        return a.byte()

    for dtype in [torch.uint8, torch.int16, torch.int32, torch.int64]:
        low_enough = torch.iinfo(dtype).min <= a.min()
        high_enough = a.max() <= torch.iinfo(dtype).max
        if low_enough and high_enough:
            return a.to(dtype)

    raise ValueError(f"Could not cast dtype={a.dtype} to integer.")


def cast_tensor(a, fp_dtype=torch.float, int_dtype=torch.long, optimal_int_dtype=False):
    """Cast torch.Tensor depending on if it is a floating point or
    non-floating point tensor.
    
    :param int_dtype: torch.dtype
        The integer dtype to cast to.
    :param fp_dtype: torch.dtype
        The floating point dtype to cast to.
    :param optimal_int_dtype: bool
        If True, will cast to the smallest possible integer dtype preserving their precision.
        If False, will cast to int_dtype.
    """
    if not isinstance(a, torch.Tensor):
        return a

    if a.is_floating_point():
        return a.to(fp_dtype)
    
    if optimal_int_dtype:
        return cast_to_optimal_integer_type(a)
    else:
        return a.to(int_dtype)


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
    #TODO: handle input and output device

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


def to_flat_tensor(
        a,
        tensors_dict=None,
        flat_dict=None,
        key=None,
        concatenate=True,
        optimal_cast_int=True):
    """Convert a Tensor or any Tensor-holding object implementing
    `to_flat_tensor` or `to_dict` to a dictionary of flat, 1D Tensors
    grouped by dtype.

    This can be used for serializing Tensor-holding objects. See
    `from_flat_tensor` for reversing this operation.

    :param a: Tensor
        Torch Tensor or Tensor-holding object implementing
        `to_flat_tensor` or `to_dict`
    :param tensors_dict: Dict[Any, Tensor]
        An existing dictionary of tensors to which the flattened `a`
        should be added
    :param flat_dict: Dict
        Dictionary of metadata describing the content of `tensors_dict`
    :param key: str
        A key to be used for storing the metadata about `a` in the
        output `flat_dict`
    :param concatenate: bool
        Whether Tensors of the same dtype in the output `tensors_dict`
        should be concatenated. If not, `tensors_dict` will hold lists
        of same-dtype Tensors
    :param optimal_cast_int: bool
        Whether non-floating-point Tensors should be cast to the
        smallest possible dtype preserving their precision while
        flattening. This allows saving memory when serializing `a` into
        `tensors_dict`
    :return:
    """

    # Prepare the data structures used for holding onto flattened data
    flat_dict = {} if flat_dict is None else flat_dict
    tensors_dict = {} if tensors_dict is None else tensors_dict
    key = next_incremental_key(
        flat_dict, prefix=f"{a.__class__.__name__}_") if key is None else key

    # Flatten `torch.Tensor` objects
    if torch.is_tensor(a):

        # Flatten the tensor without copying it
        dtype = a.dtype
        shape = a.shape
        a = a.reshape(-1)

        # If the tensor holds integers, cast to the smallest possible
        # dtype preserving precision
        if (optimal_cast_int
                and not dtype.is_floating_point
                and dtype is not torch.bool):
            a = cast_to_optimal_integer_type(a)

        # Build the start and end pointers recovering the tensor data
        # from the accumulated tensors in tensors_dict
        if a.dtype in tensors_dict.keys():
            start = sum([x.numel() for x in tensors_dict[a.dtype]])
        else:
            start = 0
        end = start + a.numel()

        # Store meta info about the tensor
        flat_info = {
            'type': dtype,
            'tensor_dtype': a.dtype,
            'tensor_shape': shape,
            'ptr': {a.dtype: (start, end)}}
        flat_dict[key] = flat_info

        # Store the flattened tensor based on its dtype
        if a.dtype in tensors_dict.keys():
            tensors_dict[a.dtype].append(a)
        else:
            tensors_dict[a.dtype] = [a]

    # Flattening objects implementing `.to_dict()`
    elif (hasattr(a.__class__, 'to_dict')
              and callable(a.__class__.to_dict)):

        # Call the class's `.to_dict()`, which is expected to return a
        # dictionary of items
        item_dict = a.to_dict()
        sub_flat_dict = {}
        for item_key, item in item_dict.items():
            tensors_dict, sub_flat_dict = to_flat_tensor(
                item,
                tensors_dict=tensors_dict,
                flat_dict=sub_flat_dict,
                key=item_key,
                concatenate=False,
                optimal_cast_int=optimal_cast_int)

        # Store meta info about the flattened object
        flat_info = {'type': a.__class__, 'flat_dict': sub_flat_dict}
        flat_dict[key] = flat_info

    # Flattening objects implementing `.to_flat_tensor()`
    elif (hasattr(a.__class__, 'to_flat_tensor')
            and callable(a.__class__.to_flat_tensor)):

        # Call the class's `.to_flat_tensor()`, which is expected to
        # operate on the `tensors_dict` and produce a `flat_dict`
        # destined for the class's `.from_flat_tensor()`
        tensors_dict, sub_flat_dict = a.to_flat_tensor(
            tensors_dict=tensors_dict,
            flat_dict=None,
            key=key,
            concatenate=False,
            optimal_cast_int=optimal_cast_int)

        # Store meta info about the flattened object
        flat_info = {'type': a.__class__, 'flat_dict': sub_flat_dict}
        flat_dict[key] = flat_info

    # Nested flattening of lists of objects
    elif isinstance(a, list):
        sub_flat_dict = {}
        for item_key, item in enumerate(a):
                tensors_dict, sub_flat_dict = to_flat_tensor(
                item,
                tensors_dict=tensors_dict,
                flat_dict=sub_flat_dict,
                key=str(item_key),
                concatenate=False,
                optimal_cast_int=optimal_cast_int)

        # Store meta info about the flattened object
        flat_info = {'type': a.__class__, 'flat_dict': sub_flat_dict}
        flat_dict[key] = flat_info

    # Other objects are assumed to not hold any `torch.Tensor`
    # object, or at least none that should be considered. These
    # will be stored as is with type 'None'
    else:
        flat_info = {'type': None, 'data': a}
        flat_dict[key] = flat_info

    # Concatenate tensors
    # IMPORTANT: concatenation does not share memory, which means
    # `to_flat_tensor(concatenate=True)` will make a copy of
    # all `torch.Tensor`s. In some situations, we may want to return
    # the list of flat tensors rather than concatenating them right
    # away. This is typically the case when nested calls to the function
    # are made by classes implementing `.to_flat_tensor()` and
    # `.from_flat_tensor()`
    if concatenate:
        tensors_dict = {k: torch.cat(v) for k, v in tensors_dict.items()}

    return tensors_dict, flat_dict


def from_flat_tensor(tensors_dict, flat_dict, key=None):
    """Reconstruct flattened tensors created using `to_flat_tensor()`.

    If the flattened data contains instances of classes implementing
    `.to_flat_tensor()` and `.from_flat_tensor()`, these will also be
    reconstructed by calling their `.from_flat_tensor()` method.
    """
    assert isinstance(tensors_dict, dict)
    assert isinstance(flat_dict, dict)

    # Prepare the keys to be recovered
    if key is None:
        keys = flat_dict.keys()
    elif isinstance(key, str):
        keys = [key]
    else:
        keys = key

    # Loop through the keys and reconstruct the corresponding Tensors or
    # objects
    out = {}
    for key in keys:

        # Recover the dictionary of meta information for the item at
        # hand
        flat_info = flat_dict[key]
        item_type = flat_info.get('type', None)

        # Objects that were stored with type 'None' or without any
        # 'type' at all have not been flattened and can be restored
        # directly from the expected 'data' key
        if item_type is None:
            out[key] = flat_info['data']
            continue

        # Reconstructing a torch.Tensor
        if isinstance(item_type, torch.dtype):
            dtype = item_type
            tensor_dtype = flat_info['tensor_dtype']
            shape = flat_info['tensor_shape']
            ptr = flat_info['ptr']
            tensor = tensors_dict[tensor_dtype]
            start = ptr[tensor_dtype][0]
            end = ptr[tensor_dtype][1]
            if torch.is_tensor(tensor):
                out[key] = tensor[start:end].to(dtype).view(shape)
            else:
                offset = 0
                for i, t in enumerate(tensor):
                    if start == offset:
                        break
                    offset += t.numel()
                if tensor[i].numel() != end - start:
                    raise ValueError
                out[key] = tensor[i].to(dtype).view(shape)
            continue

        # Reconstructing objects implementing `.from_flat_tensor()`
        if (isinstance(item_type, type)
                and hasattr(item_type, 'from_flat_tensor')
                and callable(item_type.from_flat_tensor)):
            cls = item_type
            sub_flat_dict = flat_info['flat_dict']
            out[key] = cls.from_flat_tensor(tensors_dict, sub_flat_dict)
            continue

        # Reconstructing list of objects
        if isinstance(item_type, type) and item_type is list:

            # Load all the items stored in the flat_dict, assuming these
            # relate to the object at hand
            item_dict = from_flat_tensor(tensors_dict, flat_info['flat_dict'])

            # Check expected keys are in the item_dict
            num_items, max_inc, all_inc_used = check_incremental_keys(item_dict)
            data_keys = [str(i) for i in range(num_items)]
            assert all_inc_used

            # Instantiate the object with the restored attributes
            out[key] = [item_dict[k] for k in data_keys]
            continue

        raise NotImplementedError(f"Cannot reconstruct type '{item_type}'")

    # If the read object is a dictionary holding onto  one key, we may
    # rather want to get the value itself rather than a value

    return out
