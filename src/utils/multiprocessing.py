import multiprocessing
from itertools import repeat


__all__ = ['starmap_with_kwargs']


def starmap_with_kwargs(fn, args_iter, kwargs_iter, processes=4):
    """By default, starmap only accepts args and not kwargs. This is a
    helper to get around this problem.

    :param fn: callable
        The function to starmap
    :param args_iter: iterable
        Iterable of the args
    :param kwargs_iter: iterable or dict
        Kwargs for `fn`. If an iterable is passed, the corresponding
        kwargs will be passed to each process. If a dictionary is
        passed, these same kwargs will be repeated and passed to all
        processes. NB: this behavior only works for kwargs, if the same
        args need to be passed to the `fn`, the adequate iterable must
        be passed as input
    :param processes: int
        Number of processes
    :return:
    """
    # Prepare kwargs
    if kwargs_iter is None:
        kwargs_iter = repeat({})
    if isinstance(kwargs_iter, dict):
        kwargs_iter = repeat(kwargs_iter)

    # Apply fn in multiple processes
    with multiprocessing.get_context("spawn").Pool(processes=processes) as pool:
        args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
        out = pool.starmap(apply_args_and_kwargs, args_for_starmap)

    return out

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)
