import torch
import copy
from typing import List, Dict, Union, Callable, Optional
from torch_geometric.data.storage import recursive_apply_, recursive_apply

from src.utils.memory import human_readable_memory


__all__ = ['TensorHolderMixIn']


class TensorHolderMixIn:
    """MixIn class that allows applying several torch.Tensor routines
    at the level of an object holding Tensors.

    To inherit from this mixin, child classes MUST implement:
        - `self._items()`


    This class is widely inspired from the behavior of PyG's Data and
    Storage objects:
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/data.py
    https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/data/storage.py
    """

    def _items(self) -> Dict:
        """Return a dictionary containing all Tensor and Tensor-holding
        object attributes.
        """
        raise NotImplementedError

    def items(self, *args: List[str]) -> Dict:
        """Return a dictionary containing all Tensor and Tensor-holding
        object attributes, or only the ones given in :obj:`*args`.
        """
        items = self._items()
        if args is not None and len(args) > 0:
            for key in args:
                if key not in items.keys():
                    raise ValueError(
                        f"{self.__class__.__name__} object has no '{key}'"
                        f"attribute")
            items = {k: items[k] for k in args}
        return items

    def to_dict(self):
        """Return a dictionary containing all Tensor and Tensor-holding
        object attributes.
        """
        return self.items()

    def apply_(self, func: Callable, *args: List[str]):
        """Applies the in-place function :obj:`func`, either to all
        attributes or only the ones given in :obj:`*args`.
        """
        items = self.items(*args)
        for value in items.values():
            recursive_apply_(value, func)
        return self

    def apply(self, func: Callable, *args: List[str]):
        """Applies the function :obj:`func`, either to all attributes or
        only the ones given in :obj:`*args`.
        """
        items = self.items(*args)
        for key, value in items.items():
            setattr(self, key, recursive_apply(value, func))
        return self

    def clone(self, *args: List[str]):
        """Performs cloning of tensors, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return copy.deepcopy(self).apply(lambda x: x.clone(), *args)

    def contiguous(self, *args: List[str]):
        """Ensures a contiguous memory layout, either for all attributes
        or only the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.contiguous(), *args)

    def to(
            self,
            device: Union[int, str],
            *args: List[str],
            non_blocking: bool = False,
    ):
        """Performs tensor device conversion, either for all attributes
        or only the ones given in :obj:`*args`.
        """
        return self.apply(
            lambda x: x.to(device=device, non_blocking=non_blocking), *args)

    def cpu(self, *args: List[str]):
        """Copies attributes to CPU memory, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.cpu(), *args)

    def cuda(
            self,
            device: Optional[Union[int, str]] = None,
            *args: List[str],
            non_blocking: bool = False,
    ):
        """Copies attributes to CUDA memory, either for all attributes
        or only the ones given in :obj:`*args`.
        """
        # Some PyTorch tensor like objects require a default value for `cuda`:
        device = 'cuda' if device is None else device
        return self.apply(
            lambda x: x.cuda(device, non_blocking=non_blocking),
            *args)

    def pin_memory(self, *args: List[str]):
        """Copies attributes to pinned memory, either for all attributes
        or only the ones given in :obj:`*args`.
        """
        return self.apply(lambda x: x.pin_memory(), *args)

    def share_memory_(self, *args: List[str]):
        """Moves attributes to shared memory, either for all attributes
        or only the ones given in :obj:`*args`.
        """
        return self.apply_(lambda x: x.share_memory_(), *args)

    def detach_(self, *args: List[str]):
        """Detaches attributes from the computation graph, either for
        all attributes or only the ones given in :obj:`*args`.
        """
        return self.apply_(lambda x: x.detach_(), *args)

    def detach(self, *args: List[str]):
        """Detaches attributes from the computation graph by creating a
        new tensor, either for all attributes or only the ones given in
        :obj:`*args`.
        """
        return self.apply(lambda x: x.detach(), *args)

    def requires_grad_(self, *args: List[str], requires_grad: bool = True):
        """Tracks gradient computation, either for all attributes or
        only the ones given in :obj:`*args`.
        """
        return self.apply_(
            lambda x: x.requires_grad_(requires_grad=requires_grad), *args)

    def record_stream(self, stream: torch.cuda.Stream, *args: List[str]):
        """Ensures that the tensor memory is not reused for another
        tensor until all current work queued on :obj:`stream` has been
        completed, either for all attributes or only the ones given in
        :obj:`*args`.
        """
        return self.apply_(lambda x: x.record_stream(stream), *args)

    @property
    def is_cuda(self) -> bool:
        """Returns :obj:`True` if any :class:`torch.Tensor` attribute is
        stored on the GPU, :obj:`False` otherwise.
        """
        for value in self.items().values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                return True
            if hasattr(value, 'is_cuda') and value.is_cuda:
                return True
        return False

    @property
    def device(self) -> torch.device:
        """Returns the :obj:`device` of the first :class:`torch.Tensor`
        attribute encountered in `self.items()`.
        """
        for value in self.items().values():
            if isinstance(value, torch.Tensor):
                return value.device
            if hasattr(value, 'device'):
                return value.device
        return torch.device('cpu')

    @property
    def nbytes(self):
        """Returns the number of bytes consumed by all the Tensors held
         in the object.

         Specifically, this is the sum `tensor.nbytes` for al Tensors in
        the object.
        """
        nbytes = 0
        for _, item in self.items().items():
            if torch.is_tensor(item):
                nbytes += item.nbytes
                continue
            try:
                nbytes += item.nbytes
            except AttributeError:
                pass
        return nbytes

    def print_memory_summary(self, depth=0, indent=2, prefix=''):
        """Print a summary of the memory usage for tensors inside the
        object. This is a detailed breakdown of what `self.nbytes`
        returns.
        """
        print(f"{prefix} {self.__class__.__name__}: {human_readable_memory(self.nbytes)}")
        white_length = max([len(str(key)) for key, _ in self.items().items()])
        for key, item in self.items().items():
            prefix_key = f"{'' * len(prefix)}{' ' * indent * (depth + 1)} {key:<{white_length}}"
            if torch.is_tensor(item):
                print(f"{prefix_key}: {human_readable_memory(item.nbytes)}  {tuple(item.shape)}  {item.dtype}")
                continue
            try:
                item.print_memory_summary(
                    depth=depth + 1, indent=indent, prefix=f"{prefix_key}")
            except AttributeError:
                print(f"{prefix_key}: ignored")
