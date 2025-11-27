import torch
from torch_geometric.data.storage import recursive_apply
from src.data import Data, NAG, CSRData
from src.data.tensor_holder import TensorHolderMixIn
from src.transforms import Transform
from src.utils import tensor_idx, to_float_rgb, to_byte_rgb, dropout, \
    sanitize_keys, cast_tensor, fill_list_with_string_indexing, string_to_dtype


__all__ = [
    'DataToNAG',
    'NAGToData',
    'Cast',
    'NAGCast',
    'RemoveKeys',
    'NAGRemoveKeys',
    'AddKeysTo',
    'NAGAddKeysTo',
    'NAGSelectByKey',
    'SelectColumns',
    'NAGSelectColumns',
    'DropoutColumns',
    'NAGDropoutColumns',
    'DropoutRows',
    'NAGDropoutRows',
    'NAGJitterKey']

import time

class DataToNAG(Transform):
    """Convert Data to a single-level NAG."""

    _IN_TYPE = Data
    _OUT_TYPE = NAG
    
    def __init__(self, i_level = 0):
        self.i_level = i_level

    def _process(self, data):
        return NAG([data], start_i_level = self.i_level)


class NAGToData(Transform):
    """Convert a single-level NAG to Data."""

    _IN_TYPE = NAG
    _OUT_TYPE = Data

    def _process(self, nag):
        assert nag.num_levels == 1
        return nag[0]


class Cast(Transform):
    """Cast Data attributes to the provided integer and floating point
    dtypes. In case 'rgb' or 'mean_rgb' is found, `rgb_to_float` will
    decide whether it should be cast to 'fp_dtype' or 'uint8'.
    
    If pos_dtype is provided, it will be used to cast the 'pos' attribute (instead of 'fp_dtype')
    """

    def __init__(
            self,
            fp_dtype=torch.float,
            int_dtype=torch.long,
            pos_dtype=None,
            rgb_to_float=True,
            optimal_int_dtype=False,
            ):

        if isinstance(fp_dtype, str):
            fp_dtype = string_to_dtype(fp_dtype)
        if isinstance(int_dtype, str):
            int_dtype = string_to_dtype(int_dtype)
            
        self.fp_dtype = fp_dtype
        self.int_dtype = int_dtype
        self.pos_dtype = pos_dtype
        self.rgb_to_float = rgb_to_float
        self.optimal_int_dtype = optimal_int_dtype
        
    def _process(self, data):
        # Isolate the tensor casting function as a lambda function to be
        # passed to torch_geometric's recursive_apply later on. This will
        # allow seamlessly running the casting on basic data structures
        # holding Tensors
        cast = lambda x: cast_tensor(
            x, fp_dtype=self.fp_dtype, int_dtype=self.int_dtype, optimal_int_dtype=self.optimal_int_dtype)

        for k in data.keys:

            # Specific behavior for casting 'rgb' and 'mean_rgb'
            # attributes
            if k in ['rgb', 'mean_rgb']:
                data[k] = to_float_rgb(data[k]).to(self.fp_dtype) \
                    if self.rgb_to_float else to_byte_rgb(data[k])
                continue

            # Specific behavior for 'pos_offset', which is used for
            # keeping high-precision coordinates in conjunction with
            # 'pos'. In this specific case, we prefer leaving
            # the dtype of 'pos_offset' untouched
            if k == 'pos_offset':
                continue

            if k == 'pos':
                if self.pos_dtype is not None:
                    data[k] = data[k].to(self.pos_dtype)
                else:
                    data[k] = data[k].to(self.fp_dtype)
                continue

            # Cast TensorHolderMixIn attributes such as Cluster,
            # InstanceData, etc
            if isinstance(data[k], TensorHolderMixIn):
                data[k] = data[k].apply(cast)
                continue

            # Cast Tensor attributes and native python objects holding
            # Tensors.
            # NB: we could directly use the Data.apply() mechanism for
            # this but since we also need to deal with the above edge
            # cases for 'rgb' and TensorHolderMixIn attributes, we do
            # this instead
            data[k] = recursive_apply(data[k], cast)

        # Need to additionally process attributes allowing to maintain
        # the batching mechanism throughout serialization
        for key in ['_slice_dict', '_inc_dict', '_num_graphs']:
            attr = getattr(data, key, None)
            if attr is None:
                continue
            setattr(data, key, recursive_apply(attr, cast))

        return data


class NAGCast(Cast):
    """Cast NAG attributes to the provided integer and floating point
    dtypes. In case 'rgb' or 'mean_rgb' is found and is not a floating
    point tensor, `rgb_to_float` will decide whether it should be cast
    to floats.
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def _process(self, nag):
        for i_level_relative in range(nag.num_levels):
            nag._list[i_level_relative] = super()._process(nag._list[i_level_relative])
        return nag


class RemoveKeys(Transform):
    """Remove attributes of a Data object based on their name.

    :param keys: str of list(str)
        List of attribute names
    :param strict: bool
        If True, will raise an exception if an attribute from key is
        not within the input Data keys
    """

    _NO_REPR = ['strict']

    def __init__(self, keys=None, strict=False):
        self.keys = sanitize_keys(keys, default=[])
        self.strict = strict

    def _process(self, data):
        keys = set(data.keys)
        for k in self.keys:
            if k not in keys and self.strict:
                raise Exception(f"key: {k} is not within Data keys: {keys}")
        for k in self.keys:
            delattr(data, k)
        return data


class NAGRemoveKeys(Transform):
    """Remove attributes of a NAG object based on their name.

    :param level: int or str
        Level at which to remove attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param keys: str or list(str)
        List of attribute names
    :param strict: bool=False
        If True, will raise an exception if an attribute from key is
        not within the input Data keys
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(self, level='all', keys=None, strict=False):
        assert isinstance(level, (int, str))
        self.level = level
        self.keys = sanitize_keys(keys, default=[])
        self.strict = strict

    def _process(self, nag):

        level_keys = fill_list_with_string_indexing(
            level=self.level,
            default=[],
            value=self.keys,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        transforms = [
            RemoveKeys(keys=k, strict=self.strict) for k in level_keys]

        nag.apply_data_transform(transforms)
        
        return nag


class AddKeysTo(Transform):
    """Get attributes from their keys and concatenate them to x.

    :param keys: str or list(str)
        The feature concatenated to 'to'
    :param to: str
        Destination attribute where the features in 'keys' will be
        concatenated
    :param strict: bool
        Whether we want to raise an error if a key is not found
    :param delete_after: bool
        Whether the Data attributes should be removed once added to 'to'
    """

    _NO_REPR = ['strict']

    def __init__(self, keys=None, to='x', strict=True, delete_after=True):
        self.keys = [keys] if isinstance(keys, str) else keys
        self.to = to
        self.strict = strict
        self.delete_after = delete_after

    def _process(self, data):
        data.add_keys_to(
            keys=self.keys, 
            to=self.to, 
            strict=self.strict, 
            delete_after=self.delete_after)
        return data

class NAGAddKeysTo(Transform):
    """Get attributes from their keys and concatenate them to x.

    :param level: int or str
        Level at which to remove attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param keys: str or list(str)
        The feature concatenated to 'to'
    :param to: str
        Destination attribute where the features in 'keys' will be
        concatenated
    :param strict: bool
        Whether we want to raise an error if a key is not found
    :param delete_after: bool
        Whether the Data attributes should be removed once added to 'to'
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(
            self, level='all', keys=None, to='x', strict=True,
            delete_after=True):
        self.level = level
        self.keys = [keys] if isinstance(keys, str) else keys
        self.to = to
        self.strict = strict
        self.delete_after = delete_after

    def _process(self, nag):
        level_keys = fill_list_with_string_indexing(
            level=self.level,
            default=[],
            value=self.keys,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)
        transforms = [
            AddKeysTo(
                keys=k,
                to=self.to,
                strict=self.strict,
                delete_after=self.delete_after)
            for k in level_keys]

        nag.apply_data_transform(transforms)

        return nag


class NAGSelectByKey(Transform):
    """Select the i-level nodes based on a key. The corresponding key is
    expected to exist in the i-level attributes and should hold a 1D
    boolean mask.

    :param key: str
        Key attribute expected to be found in the input NAG's `level`.
        The `key` attribute should carry a 1D boolean mask over the
        `level` nodes
    :param level: int
        NAG level based on which to operate the selection
    :param negation: bool
        Whether the mask or its complementary should be used
    :param strict: bool
        Whether we want to raise an error if the key is not found or if
        it does not carry a 1D boolean mask
    :param delete_after: bool
        Whether the `key` attribute should be removed after selection
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    _NO_REPR = ['strict']

    def __init__(
            self, key=None, level=0, negation=False, strict=True,
            delete_after=True):
        assert key is not None
        self.key = key
        self.level = level
        self.negation = negation
        self.strict = strict
        self.delete_after = delete_after

    def _process(self, nag):
        nag.assert_level_in_nag(self.level)
        
        if self.key not in nag[self.level].keys:
            if self.strict:
                raise ValueError(
                    f'Input NAG does not have `{self.key}` attribute at '
                    f'level `{self.level}`')
            return nag

        # Read the mask
        mask = nag[self.level][self.key]

        # Ensure the mask is a boolean tensor
        dtype = mask.dtype
        if dtype != torch.bool:
            if self.strict:
                raise ValueError(
                    f'`{self.key}` attribute has dtype={dtype} but '
                    f'dtype=torch.bool was expected')
            return nag

        # Ensure the mask size matches
        expected_size = torch.Size((nag[self.level].num_nodes,))
        actual_size = mask.shape
        if expected_size != actual_size:
            if self.strict:
                raise ValueError(
                    f'`{self.key}` attribute has shape={actual_size} but '
                    f'shape={expected_size} was expected')
            return nag

        # Call NAG.select using the mask on the `level` nodes
        mask = ~mask if self.negation else mask
        nag = nag.select(self.level, torch.where(mask)[0])

        # Remove the key if need be
        if self.delete_after:
            nag[self.level][self.key] = None

        return nag


class SelectColumns(Transform):
    """Select columns of an attribute based on their indices.

    :param key: str
        The Data attribute whose columns should be selected
    :param idx: int, Tensor or list
        The indices of the edge features to keep. If None, this
        transform will have no effect and edge features will be left
        untouched
    """

    def __init__(self, key=None, idx=None):
        assert key is not None, f"A Data key must be specified"
        self.key = key
        self.idx = idx

    def _process(self, data):
        if self.idx is None or getattr(data, self.key, None) is None:
            return data
        idx = tensor_idx(self.idx, device=data.device)
        data[self.key] = data[self.key][:, idx]
        return data


class NAGSelectColumns(Transform):
    """Select columns of an attribute based on their indices.

    :param level: int or str
        Level at which to select attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param key: str
        The Data attribute whose columns should be selected
    :param idx: int, Tensor or list
        The indices of the edge features to keep. If None, this
        transform will have no effect and edge features will be left
        untouched
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, level='all', key=None, idx=None):
        self.level = level
        self.key = key
        self.idx = idx

    def _process(self, nag):
        level_idx = fill_list_with_string_indexing(
            default=None,
            value=self.idx,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)
        
        transforms = [SelectColumns(key=self.key, idx=idx) for idx in level_idx]

        nag.apply_data_transform(transforms)

        return nag


class DropoutColumns(Transform):
    """Randomly set a Data attribute column to 0.

    :param p: float
        Probability of a column to be dropped
    :param key: str or list(str)
        The Data attribute whose columns should be selected
    :param inplace: bool
        Whether the dropout should be performed directly on the input
        or on a copy of it
    :param to_mean: bool
        Whether the dropped values should be set to the mean of their
        corresponding column (dim=1) or to zero (default)
    """

    def __init__(self, p=0.5, key=None, inplace=False, to_mean=False):
        assert key is not None, f"A Data key or list of keys must be specified"
        if isinstance(key, str):
            key = [key]
        self.p = p
        self.key = key
        self.inplace = inplace
        self.to_mean = to_mean

    def _process(self, data):
        # Skip dropout if p <= 0
        if self.p <= 0:
            return data

        # Skip dropout if the attribute is not present in the input Data
        for key in self.key:
            if getattr(data, key, None) is None:
                return data

            # Apply dropout on each column, inplace
            data[key] = dropout(
                data[key], p=self.p, dim=1, inplace=self.inplace,
                to_mean=self.to_mean)

        return data


class NAGDropoutColumns(Transform):
    """Randomly set a Data attribute column to 0.

    :param level: int or str
        Level at which to drop columns. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param p: float
        Probability of a column to be dropped
    :param key: str
        The Data attribute whose columns should be selected
    :param inplace: bool
        Whether the dropout should be performed directly on the input
        or on a copy of it
    :param to_mean: bool
        Whether the dropped values should be set to the mean of their
        corresponding column (dim=1) or to zero (default)
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, level='all', p=0.5, key=None, inplace=False, to_mean=False):
        assert isinstance(level, int) or level == 'all' or level.endswith('-') \
               or level.endswith('+')
        self.level = level
        self.p = p
        if isinstance(key, str):
            key = [key]
        self.key = key
        self.inplace = inplace
        self.to_mean = to_mean

    def _process(self, nag):
        # Skip dropout if p <= 0
        if self.p <= 0:
            return nag

        levels = fill_list_with_string_indexing(
            level=self.level,
            default=False,
            value=True,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        transforms = []
        # levels is of length nag.absolute_num_levels
        for i_level, do_transform in enumerate(levels):
            if do_transform :
                transforms.append(
                    DropoutColumns(
                        p=self.p,
                        key=self.key,
                        inplace=self.inplace,
                        to_mean=self.to_mean))
            else :
                transforms.append(None)
                
        nag.apply_data_transform(transforms)
        
        return nag


class DropoutRows(Transform):
    """Randomly set a Data attribute rows to 0.

    :param p: float
        Probability of a row to be dropped
    :param key: str
        The Data attribute whose rows should be selected
    :param inplace: bool
        Whether the dropout should be performed directly on the input
        or on a copy of it
    :param to_mean: bool
        Whether the dropped values should be set to the mean of their
        corresponding column (dim=1) or to zero (default)
    """

    def __init__(self, p=0.5, key=None, inplace=False, to_mean=False):
        assert key is not None, f"A Data key must be specified"
        self.p = p
        self.key = key
        self.inplace = inplace
        self.to_mean = to_mean


    def _process(self, data):
        # Skip dropout if p <= 0
        if self.p <= 0:
            return data

        # Skip dropout if the attribute is not present in the input Data
        if getattr(data, self.key, None) is None:
            return data

        # Apply dropout on each column, inplace
        data[self.key] = dropout(
            data[self.key], p=self.p, dim=0, inplace=self.inplace,
            to_mean=self.to_mean)

        return data


class NAGDropoutRows(Transform):
    """Randomly set a Data attribute rows to 0.

    :param level: int or str
        Level at which to drop rows. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param p: float
        Probability of a row to be dropped
    :param key: str
        The Data attribute whose rows should be selected
    :param inplace: bool
        Whether the dropout should be performed directly on the input
        or on a copy of it
    :param to_mean: bool
        Whether the dropped values should be set to the mean of their
        corresponding column (dim=1) or to zero (default)
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(
            self, level='all', p=0.5, key=None, inplace=False, to_mean=False):
        assert isinstance(level, int) or level == 'all' or level.endswith('-') \
               or level.endswith('+')
        self.level = level
        self.p = p
        if isinstance(key, str):
            key = [key]
        self.key = key
        self.inplace = inplace
        self.to_mean = to_mean

    def _process(self, nag):
        # Skip dropout if p <= 0
        if self.p <= 0:
            return nag

        levels = fill_list_with_string_indexing(
            level=self.level,
            default=False,
            value=True,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)
        transforms = []
        
        for i_level, do_transform in enumerate(levels):
            for k in self.key:
                if do_transform and getattr(nag[i_level], k, None) is not None:
                    transforms.append(
                        DropoutRows(
                            p=self.p,
                            key=k,
                            inplace=self.inplace,
                            to_mean=self.to_mean))
                    
                else :
                    transforms.append(None)
                
        nag.apply_data_transform(transforms)

        return nag


class NAGJitterKey(Transform):
    """Add some gaussian noise to Data['key'] for all data in a NAG.

    :param key: str or list(str)
        The attribute on which to apply jittering
    :param sigma: float or List(float)
        Standard deviation of the gaussian noise. A list may be passed
        to transform NAG levels with different parameters. Passing
        sigma <= 0 will prevent any jittering
    :param trunc: float or List(float)
        Standard deviation of the gaussian noise. A list may be passed
        to transform NAG levels with different parameters. Passing
        trunc <= 0 will not truncate the normal distribution
    :param strict: bool
        Whether an error should be raised if one of the input NAG levels
        does not have `key` attribute
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, key=None, sigma=0.01, trunc=0.05, strict=False):
        assert key is not None, "A key or list of keys must be specified"
        assert isinstance(sigma, (int, float, list))
        assert isinstance(trunc, (int, float, list))
        if isinstance(key, str):
            key = [key]
        self.key = key
        self.sigma = sigma
        self.trunc = trunc
        self.strict = strict

    def _process(self, nag):
        if not isinstance(self.sigma, list):
            sigma = [self.sigma] * nag.absolute_num_levels
        else:
            sigma = self.sigma

        if not isinstance(self.trunc, list):
            trunc = [self.trunc] * nag.absolute_num_levels
        else:
            trunc = self.trunc

        for i_level in nag.level_range:

            if sigma[i_level] <= 0:
                continue


            for k in self.key:
                if getattr(nag[i_level], k, None) is None:
                    if self.strict:
                        raise ValueError(
                            f"Input data does not have any '{k} attribute")
                    else:
                        continue

                if trunc[i_level] > 0:
                    noise = torch.nn.init.trunc_normal_(
                        torch.empty_like(nag[i_level][k]),
                        mean=0.,
                        std=sigma[i_level],
                        a=-trunc[i_level],
                        b=trunc[i_level])
                else:
                    noise = torch.randn_like(
                        nag[i_level][k]) * sigma[i_level]

                nag[i_level][k] += noise

        return nag
