import gc
import torch


__all__ = ['print_memory_size', 'garbage_collection_cuda']


def print_memory_size(a):
    assert isinstance(a, torch.Tensor)
    memory = a.element_size() * a.nelement()
    if memory > 1024 * 1024 * 1024:
        print(f'Memory: {memory / (1024 * 1024 * 1024):0.3f} Gb')
        return
    if memory > 1024 * 1024:
        print(f'Memory: {memory / (1024 * 1024):0.3f} Mb')
        return
    if memory > 1024:
        print(f'Memory: {memory / 1024:0.3f} Kb')
        return
    print(f'Memory: {memory:0.3f} bytes')


def is_oom_error(exception: BaseException) -> bool:
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # Only handle OOM errors
            raise
