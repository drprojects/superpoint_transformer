from collections.abc import Iterable
import inspect


__all__ = [
    'RADIOMETRIC_FEATURES',
    'GEOMETRIC_FEATURES',
    'POINT_FEATURES',
    'SEGMENT_BASE_FEATURES',
    'SUBEDGE_FEATURES',
    'ON_THE_FLY_HORIZONTAL_FEATURES',
    'ON_THE_FLY_VERTICAL_FEATURES',
    'sanitize_keys',
    'filter_kwargs']


RADIOMETRIC_FEATURES = [
    'rgb',
    'hsv',
    'lab',
    'intensity']

GEOMETRIC_FEATURES = [
    'linearity',
    'planarity',
    'scattering',
    'verticality',
    'curvature',
    'length',
    'surface',
    'volume',
    'normal']

POINT_FEATURES = [
    'density',
    'elevation',
    'pos_room'] + GEOMETRIC_FEATURES + RADIOMETRIC_FEATURES

SEGMENT_BASE_FEATURES = [
    'log_size',
    'log_length',
    'log_surface',
    'log_volume'] + GEOMETRIC_FEATURES

SUBEDGE_FEATURES = [
    'mean_off',
    'std_off',
    'mean_dist']

SUPEREDGE_FEATURES = [
    'centroid_dir',
    'centroid_dist',
    'normal_angle',
    'log_length',
    'log_surface',
    'log_volume',
    'log_size']

ON_THE_FLY_HORIZONTAL_FEATURES = [
    'angle_source',
    'angle_target'] + SUPEREDGE_FEATURES + SUBEDGE_FEATURES

ON_THE_FLY_VERTICAL_FEATURES = SUPEREDGE_FEATURES


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

def filter_kwargs(func, kwargs):
    """Filter kwargs to only include parameters accepted by func."""
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_params}