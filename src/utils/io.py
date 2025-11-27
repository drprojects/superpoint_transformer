import os
import h5py
import torch
import socket
import numpy as np
from time import time
from datetime import datetime
from src.utils.tensor import tensor_idx, cast_numpyfy
from src.utils.sparse import dense_to_csr, csr_to_dense


__all__ = [
    'date_time_string',
    'dated_dir',
    'save_tensor',
    'load_tensor',
    'save_tensor_dict',
    'load_tensor_dict',
    'save_dense_to_csr',
    'load_csr_to_dense']


def date_time_string():
    """Returns a string holding the current date and time. Useful for
    creating an output file or directory.
    """
    date = '-'.join([
        f'{getattr(datetime.now(), x)}'
        for x in ['year', 'month', 'day']])
    time = '-'.join([
        f'{getattr(datetime.now(), x)}'
        for x in ['hour', 'minute', 'second']])
    return f'{date}_{time}'


def dated_dir(root, create=False):
    """Returns a directory path in root, named based on the current date
    and time.
    """
    dir_name = date_time_string()
    path = os.path.join(root, dir_name)
    if create and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def save_tensor(x, f, key, fp_dtype=torch.float):
    """Save torch.Tensor to HDF5 file.

    :param x: 2D torch.Tensor
    :param f: h5 file path of h5py.File or h5py.Group
    :param key: str
        h5py.Dataset key under which to save the tensor
    :param fp_dtype: torch dtype
        Data type to which floating point tensors should be cast before
        saving
    :return:
    """
    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'w') as file:
            save_tensor(x, file, key, fp_dtype=fp_dtype)
        return

    assert isinstance(x, torch.Tensor)

    d = cast_numpyfy(x, fp_dtype=fp_dtype)
    f.create_dataset(key, data=d, dtype=d.dtype)


def load_tensor(f, key=None, idx=None, non_fp_to_long=False):
    """Load torch.Tensor from an HDF5 file. See `save_tensor` for
    writing such file. Options allow reading only part of the rows.

    :param f: h5 file path of h5py.File or h5py.Group or h5py.Dataset
    :param key: str
        h5py.Dataset key under which to the tensor was saved. Must be
        provided if f is not already a h5py.Dataset object
    :param idx: int, list, numpy.ndarray, torch.Tensor
        Used to select and read only some rows of the dense tensor.
        Supports fancy indexing
    :param non_fp_to_long: bool
        By default `save_tensor()` cast all non-float tensors to the
        smallest integer dtype before saving. This allows saving memory
        and I/O bandwidth. Upon reading, `non_fp_to_long` rules
        whether these should be cast back to int64 or kept in this
        "compressed" dtype. One good reason for not doing so is to
        accelerate data loading and device transfer. To cast the tensors
        to int64 later on in the pipeline, use the `NAGCast` and `Cast`
        transforms
    :return:
    """
    if not isinstance(f, (h5py.File, h5py.Group, h5py.Dataset)):
        with h5py.File(f, 'r') as file:
            out = load_tensor(
                file,
                key=key,
                idx=idx,
                non_fp_to_long=non_fp_to_long)
        return out

    if not isinstance(f, h5py.Dataset):
        f = f[key]

    idx = tensor_idx(idx)

    if idx is None:
        x = torch.from_numpy(f[:])
    else:
        # TODO: benchmark this to double-check. Surprisingly, I think
        #  this is faster than torch.from_numpy(f[idx]) ?
        x = torch.from_numpy(f[:])[idx]

    # By default, convert int16 and int32 to int64, might cause issues
    # for tensor indexing otherwise
    if x is not None and not x.is_floating_point() and non_fp_to_long:
        x = x.long()

    return x


def save_tensor_dict(d, f, key, fp_dtype=torch.float):
    """Save torch.Tensor to HDF5 file.

    :param d: dictionary of 2D torch.Tensors
    :param f: h5 file path of h5py.File or h5py.Group
    :param key: str
        h5py.Dataset key under which to save the tensor dictionary
    :param fp_dtype: torch dtype
        Data type to which floating point tensors should be cast before
        saving
    :return:
    """
    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'w') as file:
            save_tensor_dict(d, file, key, fp_dtype=fp_dtype)
        return

    g = f.create_group(key)
    for k, v in d.items():
        if not isinstance(v, torch.Tensor):
            continue
        save_tensor(v, g, k, fp_dtype=fp_dtype)


def load_tensor_dict(f, idx=None, non_fp_to_long=False):
    """Load a dictionary of torch.Tensor from an HDF5 file.

    :param f: h5 file path of h5py.File or h5py.Group or h5py.Dataset
    :param idx: int, list, numpy.ndarray, torch.Tensor
        Used to select and read only some rows of the dense tensor.
        Supports fancy indexing
    :param non_fp_to_long: bool
        By default `save_tensor()` cast all non-float tensors to the
        smallest integer dtype before saving. This allows saving memory
        and I/O bandwidth. Upon reading, `non_fp_to_long` rules
        whether these should be cast back to int64 or kept in this
        "compressed" dtype. One good reason for not doing so is to
        accelerate data loading and device transfer. To cast the tensors
        to int64 later on in the pipeline, use the `NAGCast` and `Cast`
        transforms
    :return:
    """
    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'w') as file:
            load_tensor_dict(
                file,
                idx=idx,
                non_fp_to_long=non_fp_to_long)
        return

    return {
        k: load_tensor(
            f[k],
            key=None,
            idx=idx,
            non_fp_to_long=non_fp_to_long)
        for k in f.keys()}


def save_dense_to_csr(x, f, fp_dtype=torch.float):
    """Compress a 2D tensor with CSR format and save it in an
    already-open HDF5.

    :param x: 2D torch.Tensor
    :param f: h5 file path of h5py.File or h5py.Group
    :param fp_dtype: torch dtype
        Data type to which floating point tensors should be cast before
        saving
    :return:
    """
    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'w') as file:
            save_dense_to_csr(x, file, fp_dtype=fp_dtype)
        return

    assert isinstance(x, torch.Tensor) and x.dim() == 2

    pointers, columns, values = dense_to_csr(x)
    save_tensor(pointers, f, 'pointers', fp_dtype=fp_dtype)
    save_tensor(columns, f, 'columns', fp_dtype=fp_dtype)
    save_tensor(values, f, 'values', fp_dtype=fp_dtype)
    f.create_dataset('shape', data=np.array(x.shape))


def load_csr_to_dense(f, idx=None, non_fp_to_long=False, verbose=False):
    """Read an HDF5 file of group produced using `dense_to_csr_hdf5` and
    return the dense tensor. An optional idx can be passed to only read
    corresponding rows from the dense tensor.

    :param f: h5 file path of h5py.File or h5py.Group
    :param idx: int, list, numpy.ndarray, torch.Tensor
        Used to select and read only some rows of the dense tensor.
        Supports fancy indexing
    :param non_fp_to_long: bool
        By default `save_tensor()` cast all non-float tensors to the
        smallest integer dtype before saving. This allows saving memory
        and I/O bandwidth. Upon reading, `non_fp_to_long` rules
        whether these should be cast back to int64 or kept in this
        "compressed" dtype. One good reason for not doing so is to
        accelerate data loading and device transfer. To cast the tensors
        to int64 later on in the pipeline, use the `NAGCast` and `Cast`
        transforms
    :param verbose: bool
    :return:
    """
    KEYS = ['pointers', 'columns', 'values', 'shape']

    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'r') as file:
            out = load_csr_to_dense(
                file,
                idx=idx,
                non_fp_to_long=non_fp_to_long,
                verbose=verbose)
        return out

    assert all(k in f.keys() for k in KEYS)

    idx = tensor_idx(idx)

    if idx is None:
        start = time()
        pointers = load_tensor(f['pointers'], non_fp_to_long=True)
        columns = load_tensor(f['columns'], non_fp_to_long=True)
        values = load_tensor(f['values'], non_fp_to_long=non_fp_to_long)
        shape = load_tensor(f['shape'], non_fp_to_long=True)
        if verbose:
            print(f'load_csr_to_dense read all      : {time() - start:0.5f}s')
        start = time()
        out = csr_to_dense(pointers, columns, values, shape=shape)
        if verbose:
            print(f'load_csr_to_dense csr_to_dense  : {time() - start:0.5f}s')
        return out

    # Read only pointers start and end indices based on idx
    start = time()
    ptr_start = load_tensor(f['pointers'], idx=idx, non_fp_to_long=True)
    ptr_end = load_tensor(f['pointers'], idx=idx + 1, non_fp_to_long=True)
    if verbose:
        print(f'load_csr_to_dense read ptr      : {time() - start:0.5f}s')

    # Create the new pointers
    start = time()
    pointers = torch.cat([
        torch.zeros(1, dtype=ptr_start.dtype),
        torch.cumsum(ptr_end - ptr_start, 0)])
    if verbose:
        print(f'load_csr_to_dense pointers      : {time() - start:0.5f}s')

    # Create the indexing tensor to select and order values.
    # Simply, we could have used a list of slices but we want to
    # avoid for loops and list concatenations to benefit from torch
    # capabilities
    start = time()
    sizes = pointers[1:] - pointers[:-1]
    val_idx = torch.arange(pointers[-1])
    val_idx -= torch.arange(pointers[-1] + 1)[
        pointers[:-1]].repeat_interleave(sizes)
    val_idx += ptr_start.repeat_interleave(sizes)
    if verbose:
        print(f'load_csr_to_dense val_idx       : {time() - start:0.5f}s')

    # Read the columns and values, now we have computed the val_idx.
    # Make sure to update the output shape too, since the rows have been
    # indexed
    start = time()
    columns = load_tensor(f['columns'], idx=val_idx, non_fp_to_long=True)
    values = load_tensor(f['values'], idx=val_idx, non_fp_to_long=non_fp_to_long)
    shape = load_tensor(f['shape'], non_fp_to_long=True)
    shape[0] = idx.shape[0]
    if verbose:
        print(f'load_csr_to_dense read values   : {time() - start:0.5f}s')

    start = time()
    out = csr_to_dense(pointers, columns, values, shape=shape)
    if verbose:
        print(f'load_csr_to_dense csr_to_dense  : {time() - start:0.5f}s')

    return out
