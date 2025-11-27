__all__ = ['next_incremental_key', 'check_incremental_keys']


def next_incremental_key(d, prefix=None):
    """Creates the next incremental string key for storing objects in
    the `d` dictionary using keys with the following structure
    "<prefix><increment>".
    """
    prefix = '' if prefix is None else prefix

    used_increments = [
        int(k.strip(prefix)) for k in d.keys()
        if isinstance(k, str)
           and k.startswith(prefix)
           and k.strip(prefix).isdigit()]
    increment = 0 if len(used_increments) == 0 else max(used_increments) + 1

    return f"{prefix}{increment}"


def check_incremental_keys(d, prefix=None):
    """Looks for keys with the structure "<prefix><increment>" in the
    `d` dictionary and returns some info about these: maximum increment
    value, number of increment values, and whether all consecutive
    increments are used.
    """
    prefix = '' if prefix is None else prefix

    used_increments = [
        int(k.strip(prefix)) for k in d.keys()
        if isinstance(k, str)
           and k.startswith(prefix)
           and k.strip(prefix).isdigit()]

    num_increments = len(used_increments)
    max_increment = max(used_increments) if num_increments > 0 else -1
    all_used = num_increments == max_increment + 1

    return num_increments, max_increment, all_used
