from collections.abc import Iterable


__all__ = [
    'POINT_FEATURES', 'SEGMENT_BASE_FEATURES', 'SUBEDGE_FEATURES',
    'ON_THE_FLY_HORIZONTAL_FEATURES', 'ON_THE_FLY_VERTICAL_FEATURES',
    'sanitize_keys']


POINT_FEATURES = [
    'rgb',
    'hsv',
    'lab',
    'density',
    'linearity',
    'planarity',
    'scattering',
    'verticality',
    'elevation',
    'normal',
    'length',
    'surface',
    'volume',
    'curvature',
    'intensity',
    'pos_room']

SEGMENT_BASE_FEATURES = [
    'linearity',
    'planarity',
    'scattering',
    'verticality',
    'curvature',
    'log_length',
    'log_surface',
    'log_volume',
    'normal',
    'log_size']

SUBEDGE_FEATURES = [
    'mean_off',
    'std_off',
    'mean_dist']

ON_THE_FLY_HORIZONTAL_FEATURES = [
    'mean_off',
    'std_off',
    'mean_dist',
    'angle_source',
    'angle_target',
    'centroid_dir',
    'centroid_dist',
    'normal_angle',
    'log_length',
    'log_surface',
    'log_volume',
    'log_size']

ON_THE_FLY_VERTICAL_FEATURES = [
    'centroid_dir',
    'centroid_dist',
    'normal_angle',
    'log_length',
    'log_surface',
    'log_volume',
    'log_size']


def sanitize_keys(keys, default=[]):
    """Sanitize an iterable of string key into a sorted list of unique
    keys. This is necessary for consistently hashing key list arguments
    of some transforms.
    """
    # Convert to list of keys
    if isinstance(keys, str):
        out = [keys]
    elif isinstance(keys, Iterable):
        out = list(keys)
    else:
        out = list(default)

    assert all(isinstance(x, str) for x in out), \
        f"Input 'keys' must be a string or an iterable of strings, but some " \
        f"non-string elements were found in '{keys}'"

    # Remove duplicates and sort elements
    out = tuple(sorted(list(set(out))))

    return out
