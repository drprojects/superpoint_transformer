__all__ = ['listify', 'listify_with_reference']


def listify(obj):
    """Convert `obj` to nested lists.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if not hasattr(obj, '__len__'):
        return obj
    if hasattr(obj, 'dim') and obj.dim() == 0:
        return obj
    if len(obj) == 0:
        return obj
    return [listify(x) for x in obj]


def listify_with_reference(arg_ref, *args):
    """listify `arg_ref` and the `args`, while ensuring that the length
    of `args` match the length of `arg_ref`. This is typically needed
    for parsing the input arguments of a function from an OmegaConf.
    """
    arg_ref = listify(arg_ref)
    args_out = [listify(a) for a in args]

    if arg_ref is None:
        return [], *([] for _ in args)

    if not isinstance(arg_ref, list):
        return [arg_ref], *[[a] for a in args_out]

    if len(arg_ref) == 0:
        return [], *([] for _ in args)

    for i, a in enumerate(args_out):
        if not isinstance(a, list):
            a = [a]
        if len(a) != len(arg_ref):
            a = a * len(arg_ref)
        args_out[i] = a

    return arg_ref, *args_out
