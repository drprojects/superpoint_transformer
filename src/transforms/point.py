import torch
import numpy as np
from sklearn.linear_model import RANSACRegressor
import pgeof
from src.utils import rgb2hsv, rgb2lab, sizes_to_pointers, to_float_rgb, \
    POINT_FEATURES, sanitize_keys
from src.transforms import Transform
from src.data import NAG


__all__ = [
    'PointFeatures', 'GroundElevation', 'RoomPosition', 'ColorAutoContrast',
    'NAGColorAutoContrast', 'ColorDrop', 'NAGColorDrop', 'ColorNormalize',
    'NAGColorNormalize']


class PointFeatures(Transform):
    """Compute pointwise features based on what is already available in
    the Data object.

    All local geometric features assume the input ``Data`` has a
    ``neighbors`` attribute, holding a ``(num_nodes, k)`` tensor of
    indices. All k neighbors will be used for local geometric features
    computation, unless some are missing (indicated by -1 indices). If
    the latter, only positive indices will be used.

    The supported feature keys are the following:
      - rgb: RGB color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
      - hsv: HSV color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
      - lab: LAB color. Assumes Data.rgb holds either [0, 1] floats or
        [0, 255] integers
      - density: local density. Assumes ``Data.neighbor_index`` and
        ``Data.neighbor_distance``
      - linearity: local linearity. Assumes ``Data.neighbor_index``
      - planarity: local planarity. Assumes ``Data.neighbor_index``
      - scattering: local scattering. Assumes ``Data.neighbor_index``
      - verticality: local verticality. Assumes ``Data.neighbor_index``
      - normal: local normal. Assumes ``Data.neighbor_index``
      - length: local length. Assumes ``Data.neighbor_index``
      - surface: local surface. Assumes ``Data.neighbor_index``
      - volume: local volume. Assumes ``Data.neighbor_index``
      - curvature: local curvature. Assumes ``Data.neighbor_index``

    :param keys: List(str)
        Features to be computed. Attributes will be saved under `<key>`
    :param k_min: int
        Minimum number of neighbors to consider for geometric features
        computation. Points with less than k_min neighbors will receive
        0-features. Assumes ``Data.neighbor_index``.
    :param k_step: int
        Step size to take when searching for the optimal neighborhood
        size following:
        http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf
        If k_step < 1, the optimal neighborhood will be computed based
        on all the neighbors available for each point.
    :param k_min_search: int
        Minimum neighborhood size used when searching the optimal
        neighborhood size. It is advised to use a value of 10 or higher.
    :param overwrite: bool
        When False, attributes of the input Data which are in `keys`
        will not be updated with the here-computed features. An
        exception to this rule is 'rgb' for which we always enforce
        [0, 1] float encoding
    """

    def __init__(
            self,
            keys=None,
            k_min=5,
            k_step=-1,
            k_min_search=25,
            overwrite=True):
        self.keys = sanitize_keys(keys, default=POINT_FEATURES)
        self.k_min = k_min
        self.k_step = k_step
        self.k_min_search = k_min_search
        self.overwrite = overwrite

    def _process(self, data):
        assert data.has_neighbors, \
            "Data is expected to have a 'neighbor_index' attribute"
        assert data.num_nodes < np.iinfo(np.uint32).max, \
            "Too many nodes for `uint32` indices"
        assert data.neighbor_index.max() < np.iinfo(np.uint32).max, \
            "Too high 'neighbor_index' indices for `uint32` indices"

        # Build the set of keys that must be computed/updated. In
        # particular, if `overwrite=False`, we do not modify
        # already-existing keys in the input Data. Except for 'rgb', for
        # which we always enforce [0, 1] float encoding
        keys = set(self.keys) if self.overwrite \
            else set(self.keys) - set(data.keys)

        # Add RGB to the features. If colors are stored in int, we
        # assume they are encoded in  [0, 255] and normalize them.
        # Otherwise, we assume they have already been [0, 1] normalized
        # NB: we ignore 'overwrite' for this key
        if 'rgb' in self.keys and data.rgb is not None:
            data.rgb = to_float_rgb(data.rgb)

        # Add HSV to the features. If colors are stored in int, we
        # assume they are encoded in  [0, 255] and normalize them.
        # Otherwise, we assume they have already been [0, 1] normalized.
        # Note: for all features to live in a similar range, we
        # normalize H in [0, 1]
        if 'hsv' in keys and data.rgb is not None:
            hsv = rgb2hsv(to_float_rgb(data.rgb))
            hsv[:, 0] /= 360.
            data.hsv = hsv

        # Add LAB to the features. If colors are stored in int, we
        # assume they are encoded in  [0, 255] and normalize them.
        # Otherwise, we assume they have already been [0, 1] normalized.
        # Note: for all features to live in a similar range, we
        # normalize L in [0, 1] and ab in [-1, 1]
        if 'lab' in keys and data.rgb is not None:
            data.lab = rgb2lab(to_float_rgb(data.rgb)) / 100

        # Add local surfacic density to the features. The local density
        # is approximated as K / D² where K is the number of nearest
        # neighbors and D is the distance of the Kth neighbor. We
        # normalize by D² since points roughly lie on a 2D manifold.
        # Note that this takes into account partial neighborhoods where
        # -1 indicates absent neighbors
        if 'density' in keys:
            dmax = data.neighbor_distance.max(dim=1).values
            k = data.neighbor_index.ge(0).sum(dim=1)
            data.density = (k / dmax ** 2).view(-1, 1)

        # Add local geometric features
        needs_geof = any((
            'linearity' in keys,
            'planarity' in keys,
            'scattering' in keys,
            'verticality' in keys,
            'normal' in keys))
        if needs_geof and data.pos is not None:

            # Prepare data for numpy boost interface. Note: we add each
            # point to its own neighborhood before computation
            device = data.pos.device
            xyz = data.pos.cpu().numpy()
            nn = torch.cat(
                (torch.arange(xyz.shape[0]).view(-1, 1), data.neighbor_index),
                dim=1)
            k = nn.shape[1]

            # Check for missing neighbors (indicated by -1 indices)
            n_missing = (nn < 0).sum(dim=1)
            if (n_missing > 0).any():
                sizes = k - n_missing
                nn = nn[nn >= 0]
                nn_ptr = sizes_to_pointers(sizes.cpu())
            else:
                nn = nn.flatten().cpu()
                nn_ptr = torch.arange(xyz.shape[0] + 1) * k
            nn = nn.numpy().astype('uint32')
            nn_ptr = nn_ptr.numpy().astype('uint32')

            # Make sure array are contiguous before moving to C++
            xyz = np.ascontiguousarray(xyz)
            nn = np.ascontiguousarray(nn)
            nn_ptr = np.ascontiguousarray(nn_ptr)

            # C++ geometric features computation on CPU
            if self.k_step < 0:
                f = pgeof.compute_features(
                    xyz, 
                    nn, 
                    nn_ptr, 
                    self.k_min, 
                    verbose=False)
            else:
                f = pgeof.compute_features_optimal(
                    xyz,
                    nn,
                    nn_ptr,
                    self.k_min,
                    self.k_step,
                    self.k_min_search,
                    verbose=False)
            f = torch.from_numpy(f)

            # Keep only required features
            if 'linearity' in keys:
                data.linearity = f[:, 0].view(-1, 1).to(device)

            if 'planarity' in keys:
                data.planarity = f[:, 1].view(-1, 1).to(device)

            if 'scattering' in keys:
                data.scattering = f[:, 2].view(-1, 1).to(device)

            # Heuristic to increase importance of verticality in
            # partition
            if 'verticality' in keys:
                data.verticality = f[:, 3].view(-1, 1).to(device)
                data.verticality *= 2

            if 'curvature' in keys:
                data.curvature = f[:, 10].view(-1, 1).to(device)

            if 'length' in keys:
                data.length = f[:, 7].view(-1, 1).to(device)

            if 'surface' in keys:
                data.surface = f[:, 8].view(-1, 1).to(device)

            if 'volume' in keys:
                data.volume = f[:, 9].view(-1, 1).to(device)

            # As a way to "stabilize" the normals' orientation, we
            # choose to express them as oriented in the z+ half-space
            if 'normal' in keys:
                data.normal = f[:, 4:7].view(-1, 3).to(device)
                data.normal[data.normal[:, 2] < 0] *= -1

        return data


class GroundElevation(Transform):
    """Compute pointwise elevation by approximating the ground as a
    plane using RANSAC.

    Parameters
    ----------
    :param threshold: float
        Ground points will be searched within `threshold` of the lowest
        point in the cloud. Adjust this if the lowest point is below the
        ground or if you have large above-ground planar structures
    :param scale: float
        Scaling by which the computed elevation should be divided, for
        the sake of normalization
    """

    def __init__(self, threshold=1.5, scale=3.0):
        self.threshold = threshold
        self.scale = scale

    def _process(self, data):
        # Recover the point positions
        pos = data.pos.cpu().numpy()

        # To avoid capturing high above-ground flat structures, we only
        # keep points which are within `threshold` of the lowest point.
        idx_low = np.where(pos[:, 2] - pos[:, 2].min() < self.threshold)[0]

        # Search the ground plane using RANSAC
        ransac = RANSACRegressor(random_state=0, residual_threshold=1e-3).fit(
            pos[idx_low, :2], pos[idx_low, 2])

        # Compute the pointwise elevation as the distance to the plane
        # and scale it
        h = pos[:, 2] - ransac.predict(pos[:, :2])
        h = h / self.scale

        # Save in Data attribute `elevation`
        data.elevation = torch.from_numpy(h).to(data.device).view(-1, 1)

        return data


class RoomPosition(Transform):
    """Compute the pointwise normalized room coordinates, as defined
    in the S3DIS paper section 3.2:
    https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf

    Results will be saved in the `pos_room` attribute.

    NB: this is rather suited for indoor setting, with regular
    dimensions and not so much for open outdoor clouds with unbounded
    sizes.

    Parameters
    ----------
    :param elevation: bool
        Whether the `elevation` attribute should be used to position the
        ground to z=0. If True, this assumes `GroundElevation` has been
        called previously to produce the `elevation` attribute. If
        False, it is assumed the ground/floor of the input cloud is
        already positioned at z=0
    """

    def __init__(self, elevation=False):
        self.elevation = elevation

    def _process(self, data):
        # Recover the point positions
        pos = data.pos.clone()

        # Shift ground to z=0, if required. Otherwise the ground is
        # assumed to be already at z=0
        if self.elevation:
            assert getattr(data, 'elevation', None) is not None
            pos[:, 2] -= data.elevation

        # Shift XY
        pos[:, :2] -= pos[:, :2].min(dim=0).values.view(1, -1)

        # Scale XYZ based on the maximum values. i.e. the highest point
        # will be considered as the ceiling
        pos /= pos.max(dim=0).values.view(1, -1)

        # Save in Data attribute `pos_room`
        data.pos_room = pos

        return data


class ColorTransform(Transform):
    """Parent class for color-based point Transforms, to avoid redundant
    code.

    :param x_idx: int
        If specified, the colors will be searched in
        `data.x[:, x_idx:x_idx + 3]` instead of `data.rgb`
    """
    KEYS = ['rgb', 'lab', 'hsv']

    def __init__(self, x_idx=None):
        self.x_idx = x_idx

    def _process(self, data):
        if self.x_idx is None:
            for key in self.KEYS:
                mean_key = f'mean_{key}'
                if getattr(data, key, None) is not None:
                    data[key] = self._apply_func(data[key])
                if getattr(data, mean_key, None) is not None:
                    data[mean_key] = self._apply_func(data[mean_key])

        elif self.x_idx is not None and getattr(data, 'x', None) is not None:
            data.x[:, self.x_idx:self.x_idx + 3] = self._apply_func(
                data.x[:, self.x_idx:self.x_idx + 3])

        return data

    def _apply_func(self, rgb):
        return self._func(rgb)

    def _func(self, rgb):
        raise NotImplementedError


class ColorAutoContrast(ColorTransform):
    """Apply some random contrast to the point colors.

    credit: https://github.com/guochengqian/openpoints

    :param p: float
        Probability of the transform to be applied
    :param blend: float (optional)
        Blend factor, controlling the contrasting intensity
    :param x_idx: int
        If specified, the colors will be searched in
        `data.x[:, x_idx:x_idx + 3]` instead of `data.rgb`
    """
    KEYS = ['rgb']

    def __init__(self, p=0.2, blend=None, x_idx=None):
        super().__init__(x_idx=x_idx)
        self.p = p
        self.blend = blend

    def _func(self, rgb):
        device = rgb.device

        if torch.rand(1, device=device) < self.p:

            # Compute the contrasted colors
            lo = rgb.min(dim=0).values.view(1, -1)
            hi = rgb.max(dim=0).values.view(1, -1)
            contrast_feat = (rgb - lo) / (hi - lo)

            # Blend the maximum contrast with the current color
            blend = torch.rand(1, device=device) \
                if self.blend is None else self.blend
            rgb = (1 - blend) * rgb + blend * contrast_feat

        return rgb


class NAGColorAutoContrast(ColorAutoContrast):
    """Apply some random contrast to the point colors.

    credit: https://github.com/guochengqian/openpoints

    :param level: int or str
        Level at which to remove attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param p: float
        Probability of the transform to be applied
    :param blend: float (optional)
        Blend factor, controlling the contrasting intensity
    :param x_idx: int
        If specified, the colors will be searched in
        `data.x[:, x_idx:x_idx + 3]` instead of `data.rgb`
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    KEYS = ['rgb']

    def __init__(self, *args, level='all', **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level

    def _process(self, nag):

        level_p = [-1] * nag.num_levels
        if isinstance(self.level, int):
            level_p[self.level] = self.p
        elif self.level == 'all':
            level_p = [self.p] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_p[i:] = [self.p] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_p[:i] = [self.p] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [
            ColorAutoContrast(p=p, blend=self.blend, x_idx=self.x_idx)
            for p in level_p]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag


class ColorDrop(ColorTransform):
    """Randomly set point colors to 0.

    :param p: float
        Probability of the transform to be applied
    :param x_idx: int
        If specified, the colors will be searched in
        `data.x[:, x_idx:x_idx + 3]` instead of `data.rgb`
    """

    def __init__(self, p=0.2, x_idx=None):
        super().__init__(x_idx=x_idx)
        self.p = p

    def _func(self, rgb):
        if torch.rand(1, device=rgb.device) < self.p:
            rgb *= 0
        return rgb


class NAGColorDrop(ColorDrop):
    """Randomly set point colors to 0.

    :param level: int or str
        Level at which to remove attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param p: float
        Probability of the transform to be applied
    :param x_idx: int
        If specified, the colors will be searched in
        `data.x[:, x_idx:x_idx + 3]` instead of `data.rgb`
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG

    def __init__(self, *args, level='all', **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level

    def _process(self, nag):

        level_p = [-1] * nag.num_levels
        if isinstance(self.level, int):
            level_p[self.level] = self.p
        elif self.level == 'all':
            level_p = [self.p] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_p[i:] = [self.p] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_p[:i] = [self.p] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [ColorDrop(p=p, x_idx=self.x_idx) for p in level_p]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag


class ColorNormalize(ColorTransform):
    """Normalize the colors using given means and standard deviations.

    credit: https://github.com/guochengqian/openpoints

    :param mean: list
        Channel-wise means
    :param std: list
        Channel-wise standard deviations
    :param x_idx: int
        If specified, the colors will be searched in
        `data.x[:, x_idx:x_idx + 3]` instead of `data.rgb`
    """
    KEYS = ['rgb']

    def __init__(
            self,
            mean=[0.5136457, 0.49523646, 0.44921124],
            std=[0.18308958, 0.18415008, 0.19252081],
            x_idx=None):
        super().__init__(x_idx=x_idx)
        assert all(x > 0 for x in std), "std values must be >0"
        self.mean = mean
        self.std = std

    def _func(self, rgb):
        mean = torch.as_tensor(self.mean, device=rgb.device).view(1, -1)
        std = torch.as_tensor(self.std, device=rgb.device).view(1, -1)
        rgb = (rgb - mean) / std
        return rgb


class NAGColorNormalize(ColorNormalize):
    """Normalize the colors using given means and standard deviations.

    credit: https://github.com/guochengqian/openpoints

    :param level: int or str
        Level at which to remove attributes. Can be an int or a str. If
        the latter, 'all' will apply on all levels, 'i+' will apply on
        level-i and above, 'i-' will apply on level-i and below
    :param mean: list
        Channel-wise means
    :param std: list
        Channel-wise standard deviations
    :param x_idx: int
        If specified, the colors will be searched in
        `data.x[:, x_idx:x_idx + 3]` instead of `data.rgb`
    """

    _IN_TYPE = NAG
    _OUT_TYPE = NAG
    KEYS = ['rgb']

    def __init__(self, *args, level='all', **kwargs):
        super().__init__(*args, **kwargs)
        self.level = level

    def _process(self, nag):

        level_mean = [[0, 0, 0]] * nag.num_levels
        level_std = [[1, 1, 1]] * nag.num_levels
        if isinstance(self.level, int):
            level_mean[self.level] = self.mean
            level_std[self.level] = self.std
        elif self.level == 'all':
            level_mean = [self.mean] * nag.num_levels
            level_std = [self.std] * nag.num_levels
        elif self.level[-1] == '+':
            i = int(self.level[:-1])
            level_mean[i:] = [self.mean] * (nag.num_levels - i)
            level_std[i:] = [self.std] * (nag.num_levels - i)
        elif self.level[-1] == '-':
            i = int(self.level[:-1])
            level_mean[:i] = [self.mean] * i
            level_std[:i] = [self.std] * i
        else:
            raise ValueError(f'Unsupported level={self.level}')

        transforms = [
            ColorNormalize(mean=mean, std=std, x_idx=self.x_idx)
            for mean, std in zip(level_mean, level_std)]

        for i_level in range(nag.num_levels):
            nag._list[i_level] = transforms[i_level](nag._list[i_level])

        return nag
