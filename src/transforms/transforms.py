from typing import Union, List
from torch_geometric.transforms import BaseTransform
from src.data import Data


__all__ = ['Transform']


class Transform(BaseTransform):
    """Transform on `_IN_TYPE` returning `_OUT_TYPE`."""

    _IN_TYPE = Data
    _OUT_TYPE = Data
    _NO_REPR = []

    def _process(self, x: _IN_TYPE):
        raise NotImplementedError

    def __call__(self, x: Union[_IN_TYPE, List]):
        assert isinstance(x, (self._IN_TYPE, list))
        if isinstance(x, list):
            return [self.__call__(e) for e in x]
        return self._process(x)

    @property
    def _repr_dict(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._NO_REPR}

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self._repr_dict.items()])
        return f'{self.__class__.__name__}({attr_repr})'
