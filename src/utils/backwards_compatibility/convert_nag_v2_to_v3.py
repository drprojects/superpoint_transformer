"""
Utility class to convert NAG files from version 2.1.0 to 3.0.0 format.

Main changes:
- Modified serialization behavior of data structures
- Updated HDF5 keys/schema
- Changes in how Data objects are stored within NAG

Usage:
    converter = ConvertNAG_V2ToV3(input_path="data_v2.h5")
    output_path = converter.convert()
    # or
    python -m src.utils.backwards_compatibility.convert_nag_v2_to_v3 [path/to/nag_v2.h5]
"""

import h5py
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, List
from time import time

from src.utils import (
    tensor_idx, 
    load_tensor, 
    to_float_rgb, 
    to_byte_rgb, 
    load_csr_to_dense)
from src.data import NAG
from src.data.csr import CSRData

from src.data import (
    NAG, 
    Data, 
    Cluster,
    InstanceData,
)

__all__ = ["ConvertNAG_V2ToV3"]

def load_old_nag(
            path: str,
            low: int = 0,
            high: int = -1,
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            keys_idx: List[str] = None,
            keys_low: List[str] = None,
            keys: List[str] = None,
            update_super: bool = True,
            update_sub: bool = True,
            verbose: bool = False
    ) -> 'NAG':
        """Load NAG from an HDF5 file. See `NAG.save` for writing such
        file. Options allow reading only part of the data.

        NB: if relevant, a NAGBatch object will be returned.

        :param path: str
            Path the file
        :param low: int
            Lowest partition level to read
        :param high: int
            Highest partition level to read
        :param idx: list, array, tensor, slice
            Index or boolean mask used to select from low
        :param keys_idx: list(str)
            Keys on which the indexing should be applied
        :param keys_low: list(str)
            Keys to read for low-level. If None, all keys will be read
        :param keys: list(str)
            Keys to read. If None, all keys will be read
        :param update_sub: bool
            See NAG.select and Data.select
        :param update_super:
            See NAG.select and Data.select
        :param verbose: bool
        :return:
        """
        assert idx is None, "Indexing is not supported for conversion."
        keys_low = keys if keys_low is None and keys is not None else keys_low
    

        data_list = []
        with h5py.File(path, 'r') as f:

            # Initialize partition levels min and max to read from the
            # file. This functionality is especially intended for
            # loading levels 1 and above when we want to avoid loading
            # the memory-costly level-0 points
            low = max(low, 0)
            high = len(f) - 1 if high < 0 else min(high, len(f) - 1)

            # Make sure all required partitions are present in the file
            assert all([
                f'partition_{k}' in f.keys()
                for k in range(low, high + 1)])

            # Apply index selection on the low only, if required. For
            # all subsequent levels, only keys selection is available
            for i in range(low, high + 1):
                if i == low:
                    data = load_old_data(
                        f[f'partition_{i}'], idx=idx, keys_idx=keys_idx,
                        keys=keys_low, update_sub=update_sub,
                        verbose=verbose)
                else:
                    data = load_old_data(
                        f[f'partition_{i}'], keys=keys, update_sub=False,
                        verbose=verbose)
                data_list.append(data)
                
        cls = NAG
        return cls(data_list)

def load_old_data(
    f: Union[str, h5py.File, h5py.Group],
    idx: Union[int, List, np.ndarray, torch.Tensor] = None,
    keys_idx: List[str] = None,
    keys: List[str] = None,
    update_sub: bool = True,
    verbose: bool = False,
    rgb_to_float: bool = False
    ) -> Dict[str, Any]: 
    
    if not isinstance(f, (h5py.File, h5py.Group)):
        with h5py.File(f, 'r') as file:
            out = load_old_data(
                file, 
                idx=idx, 
                keys_idx=keys_idx, 
                keys=keys,
                update_sub=update_sub, 
                verbose=verbose,
                rgb_to_float=rgb_to_float
                )
        return out
    
    # Recover the keys that do not support node indexing
    _not_indexable = [
        '_csr_', '_cluster_', '_instance_data_', 'edge_index', 'edge_attr',
        '_slice_dict', '_inc_dict', '_num_graphs']
    
    if '_not_indexable_' in f.keys():
        _not_indexable += [s.decode("utf-8") for s in f['_not_indexable_']]

        d_dict = {}
        csr_keys = []
        cluster_keys = []
        instance_data_keys = []

        # This script converts a Data object (never Batch)
        cls = Data

        # Deal with special keys first, then read other keys if required
        for k in f.keys():
            start = time()
            if k == '_not_indexable_':
                continue
            if k == '_csr_':
                csr_keys = list(f[k].keys())
                continue
            if k == '_cluster_':
                cluster_keys = list(f[k].keys())
                continue
            if k == '_instance_data_':
                instance_data_keys = list(f[k].keys())
                continue
            if k in ['_slice_dict', '_inc_dict']:
                print(f"Key '{k}' of batch has been found in the file."\
                    "But the script returns object as non-batch.")
                # if cls == Batch:
                #     d_dict[k] = load_tensor_dict(f[k]) #TODO@Geist: fix this
                continue
            if k == '_num_graphs':
                print(f"Key '{k}' of batch has been found in the file."\
                    "But the script returns object as non-batch.")
                # if cls == Batch:
                #     d_dict[k] = f['_num_graphs'][0]
                continue
            if keys is None or k in keys :
                d_dict[k] = load_tensor(f[k])
            if verbose and k in d_dict.keys():
                print(f'load_old_data {k:<22}: {time() - start:0.5f}s')

        # Small sanity check on '_slice_dict' and '_inc_dict'. It is
        # possible, for attributes of type 'other' that the _inc_dict
        # contains a 'None' value when the _slice_dict holds a Tensor.
        # As a result of calling 'save_tensor_dict()' and
        # 'load_tensor_dict()', these values will be lost. So we need to
        # restore them here
        if '_slice_dict' in d_dict.keys():
            for k in set(d_dict['_slice_dict']) - set(d_dict['_inc_dict']):
                d_dict['_inc_dict'][k] = None

        # Special key '_csr_' holds data saved in CSR format
        for k in csr_keys:
            d_dict[k] = load_csr_to_dense(f['_csr_'][k], 
                                          verbose=verbose, 
                                          non_fp_to_long=True)


        # Special key '_cluster_' holds Cluster data
        for k in cluster_keys:
            d_dict[k] = Old_Cluster.load(
                f['_cluster_'][k], update_sub=update_sub,
                verbose=verbose)[0]

        # Special key '_instance_data_' holds InstanceData data
        for k in instance_data_keys:
            d_dict[k] = Old_InstanceData.load(
                    f['_instance_data_'][k], verbose=verbose)

        # In case RGB is among the keys and is in integer type, convert
        # to float
        for k in ['rgb', 'mean_rgb']:
            if k in d_dict.keys():
                d_dict[k] = to_float_rgb(d_dict[k]) if rgb_to_float \
                    else to_byte_rgb(d_dict[k])

        return cls(**d_dict)

        
class Old_CSRData:
    __value_serialization_keys__ = None
    __pointer_serialization_key__ = 'pointers'
    __is_index_value_serialization_key__ = 'is_index_value'
    
    # Class of the codebase v3
    __output_class__ = CSRData
    
    @classmethod
    def load(cls,
            f: Union[str, h5py.File, h5py.Group],
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            verbose: bool = False
        ) -> 'CSRData':
            """Load CSRData from an HDF5 file. See `CSRData.save`
            for writing such file. Options allow reading only part of the
            clusters.

            :param f: h5 file path of h5py.File or h5py.Group
            :param idx: int, list, numpy.ndarray, torch.Tensor
                Used to select clusters when reading. Supports fancy
                indexing
            :param verbose: bool
            """

            
            assert idx is None, "Indexing is not supported during conversion."
        
            if not isinstance(f, (h5py.File, h5py.Group)):
                with h5py.File(f, 'r') as file:
                    out = Old_CSRData.load(file, idx=idx, verbose=verbose)
                return out


            # Check expected keys are in the file
            pointer_key = cls.__pointer_serialization_key__
            value_keys = cls.__value_serialization_keys__
            value_keys = value_keys if value_keys is not None else []
            is_index_value_key = cls.__is_index_value_serialization_key__
            assert pointer_key in f.keys()
            assert all(k in f.keys() for k in value_keys)
            assert is_index_value_key is None or is_index_value_key in f.keys()

            # If no value keys are provided, CSRData.save() falls back to
            # using integers to index values. So, need to infer the number
            # of values from the consecutive integer keys in the file
            if len(value_keys) == 0:
                num_values = 0
                while str(num_values) in f.keys():
                    num_values += 1
                value_keys = [str(i) for i in range(num_values)]

            if idx is None or idx.shape[0] == 0:
                pointers = load_tensor(f[pointer_key])
                values = [load_tensor(f[k]) for k in value_keys]

                # ----- Conversion code -----
                # Build of the new CSRData object
                
                if is_index_value_key is not None:
                    is_index_value = load_tensor(f[is_index_value_key]).bool()
                else:
                    is_index_value = None
                    
                out = cls.__output_class__(pointers, 
                        *values,
                        is_index_value=is_index_value)
                    
                return out

class Old_Cluster(Old_CSRData):
    __value_serialization_keys__ = ['points']
    __is_index_value_serialization_key__ = None
    
    __output_class__ = Cluster
    
    @classmethod
    def load(
                cls,
                f: Union[str, h5py.File, h5py.Group],
                idx: Union[int, List, np.ndarray, torch.Tensor] = None,
                update_sub: bool = True,
                verbose: bool = False,
                **kwargs
        ) -> Dict:
            """Load Cluster from an HDF5 file. See `Cluster.save` for
            writing such file. Options allow reading only part of the
            clusters.

            This reproduces the behavior of Cluster.select but without
            reading the full pointer data from disk.

            :param f: h5 file path of h5py.File or h5py.Group
            :param idx: int, list, numpy.ndarray, torch.Tensor
                Used to select clusters when reading. Supports fancy
                indexing
            :param update_sub: bool
                If True, the point (i.e. subpoint) indices will also be
                updated to maintain dense indices. The output will then
                contain '(idx_sub, sub_super)' which can help apply these
                changes to maintain consistency with lower hierarchy levels
                of a NAG.
            :param verbose: bool

            :return: cluster, (idx_sub, sub_super)
            """
            assert not update_sub, "We don't support update_sub for conversion."
            
            if not isinstance(f, (h5py.File, h5py.Group)):
                with h5py.File(f, 'r') as file:
                    out = Old_Cluster.load(
                        file, idx=idx, update_sub=update_sub, verbose=verbose)
                return out

            # CSRData load behavior
            out = super().load(f, idx=idx, verbose=verbose)
            cluster = out[0] if isinstance(out, tuple) else out
            
            return cluster, (None, None)

class Old_InstanceData(Old_CSRData):
    __value_keys__ = ['obj', 'count', 'y']
    __is_index_value_serialization_key__ = None
    
    __output_class__ = InstanceData
            
class ConvertNAG_V2ToV3:
    def __init__(self, input_path: str, output_path: str = None, backup: bool = True):
        self.input_path = input_path
        self.output_path = output_path
        self.backup = backup

    def get_output_path(self):
        if self.output_path is None:
            return self.input_path.replace('.h5', '_v3.h5')
        return self.output_path
    
    def convert(self):
        nag = load_old_nag(self.input_path,
                           update_sub=False,
                           update_super=False)

        nag.save(self.get_output_path())
        

if __name__ == "__main__":
    """
    Usage: python -m src.utils.backwards_compatibility.convert_nag_v2_to_v3 nag_v2.h5
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert NAG files from v2.1.0 to v3.0.0")
    parser.add_argument("input_path", type=str, help="Path to input HDF5 file")
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Path to output file (default: input_path with '_v3' suffix)")
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Don't create backup of original file")
    
    args = parser.parse_args()
    
    converter = ConvertNAG_V2ToV3(
        input_path=args.input_path,
        output_path=args.output_path,
        backup=not args.no_backup
    )
    
    converter.convert()

