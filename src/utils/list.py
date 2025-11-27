from omegaconf import ListConfig

__all__ = ['listify', 'listify_with_reference', 'fill_list_with_string_indexing']


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

    if not isinstance(arg_ref, (list, ListConfig)):
        return [arg_ref], *[[a] for a in args_out]

    if len(arg_ref) == 0:
        return [], *([] for _ in args)

    for i, a in enumerate(args_out):
        if not isinstance(a, (list, ListConfig)):
            a = [a]
        if len(a) != len(arg_ref):
            a = a * len(arg_ref)
        args_out[i] = a

    return arg_ref, *args_out

def fill_list_with_string_indexing(
        level,
        default,
        value,
        output_length,
        start_index):
    """Returns a list of length `output_length` :
    - with `value` set at the specified `level`,
    - with `default` set for the other levels.

    This is typically used to set the parameters for `Transform`s to be
    applied at different levels of the data hierarchy of a `NAG`. It is
    used before using `NAG.apply_data_transform`.
    
    We refer as 'string_indexing' the possibility to use specific strings 
    (e.g. 'all', '1+', '2-') to parametrize the transforms.

    :param level : int or string index
        Indices at which to set `value` in the output list. Can be an
        int or a str. If the latter, 'all' will apply on all levels,
        'i+' will apply on level-i and above, 'i-' will apply on
        level-i and below
    :param default : default value in the list
    :param value : value to set at the index
    :param output_length : length of the output list
    :param start_index : first index to set the value
        when level='all'

    :return output_list : list
    """
    output_list = [default] * output_length

    if isinstance(level, int):
        output_list[level] = value
    elif level == 'all':
        output_list[start_index :] = [value] * (output_length - start_index)
    elif level[-1] == '+':
        i = int(level[:-1])
        output_list[i:] = [value] * (output_length - i)
    elif level[-1] == '-':
        i = int(level[:-1])
        output_list[:i] = [value] * i
    else:
        raise ValueError(f'Unsupported level={level}')

    return output_list
