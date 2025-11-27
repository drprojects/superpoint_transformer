import torch
from time import time
from typing import Union, List
from torch_geometric.transforms import BaseTransform

import src
from src.data import Data, NAG

__all__ = ['Transform']


class Transform(BaseTransform):
    """Transform on `_IN_TYPE` returning `_OUT_TYPE`."""

    _IN_TYPE = Data
    _OUT_TYPE = Data
    _NO_REPR = []

    def _process(self, x: _IN_TYPE):
        raise NotImplementedError

    def __call__(self, x: Union[_IN_TYPE, List], verbose: bool = False):
        if verbose or src.is_debug_enabled():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time()

        if isinstance(x, list):
            return [self.__call__(e) for e in x]

        elif isinstance(x, self._IN_TYPE):
            out = self._process(x)

        # Special case: allow Data input when transform expects NAG
        # Wrap Data in NAG, process, then unwrap if needed
        elif isinstance(x, Data) and self._IN_TYPE == NAG:
            out = self._process(NAG([x], start_i_level = 0))
            out = out[0]
        else:
            raise ValueError(
                f"Expected input type {self._IN_TYPE} or List({self._IN_TYPE}) "
                f"but received {type(x)} instead")

        if verbose or src.is_debug_enabled():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f'{self.__repr__():<30}: {time() - start:0.5f}')

        return out

    @property
    def _repr_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._NO_REPR}

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self._repr_dict.items()])
        return f'{self.__class__.__name__}({attr_repr})'
