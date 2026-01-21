import os
import torch
import numpy as np

from src.utils import (
    rgb2hsv,
    rgb2lab,
    to_float_rgb,
    POINT_FEATURES,
    GEOMETRIC_FEATURES,
    sanitize_keys,
    filter_by_z_distance_of_global_min,
    filter_by_local_z_min,
    filter_by_verticality,
    single_plane_model,
    neighbor_interpolation_model,
    mlp_model,
    fill_list_with_string_indexing,
    geometric_features,
    filter_kwargs)
from src.transforms import Transform
from src.data import NAG
from src.models.components.spt import SPT

import logging
log = logging.getLogger(__name__)

__all__ = [
    'PointFeatures',
    'GroundElevation',
    'RoomPosition',
    'ColorAutoContrast',
    'NAGColorAutoContrast',
    'ColorDrop',
    'NAGColorDrop',
    'ColorNormalize',
    'NAGColorNormalize',
    'PretrainedCNN']


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
        computation. Points with fewer than k_min neighbors will receive
        0-features. Assumes ``Data.neighbor_index``.
    :param k_step: int
        Step size to take when searching for the optimal neighborhood
        size following:
        http://lareg.ensg.eu/labos/matis/pdf/articles_revues/2015/isprs_wjhm_15.pdf
        If k_step < 1, the optimal neighborhood will be computed based
        on all the neighbors available for each point
    :param k_min_search: int
        Minimum neighborhood size used when searching the optimal
        neighborhood size. It is advised to use a value of 10 or higher
    :param add_self_as_neighbor: bool
    :param chunk_size: int, float
        Allows mitigating memory use when the inputs are on GPU. If
        `chunk_size > 1`, the input point cloud will be processed into
        chunks of `chunk_size`. If `0 < chunk_size < 1`, then the point
        cloud will be divided into parts of `xyz.shape[1] * chunk_size`
        or smaller
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
            add_self_as_neighbor=True,
            chunk_size=100000,
            overwrite=True):
        self.keys = sanitize_keys(keys, default=POINT_FEATURES)
        self.k_min = k_min
        self.k_step = k_step
        self.k_min_search = k_min_search
        self.add_self_as_neighbor = add_self_as_neighbor
        self.chunk_size = chunk_size
        self.overwrite = overwrite

    def _process(self, data):
        assert data.num_nodes < np.iinfo(np.uint32).max, \
            "Too many nodes for `uint32` indices"

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
        geof_keys_needed = set(keys) & set(GEOMETRIC_FEATURES)
        if len(geof_keys_needed) > 0 and data.pos is not None:
            assert data.has_neighbors, \
                "Data is expected to have a 'neighbor_index' attribute"
            assert data.neighbor_index.max() < np.iinfo(np.uint32).max, \
                "Too high 'neighbor_index' indices for `uint32` indices"
            
            features = geometric_features(
                data.pos,
                data.neighbor_index,
                k_min=self.k_min,
                k_step=self.k_step,
                k_min_search=self.k_min_search,
                add_self_as_neighbor=self.add_self_as_neighbor,
                chunk_size=self.chunk_size)
            for key in geof_keys_needed:
                data[key] = features[key]

        return data


class GroundElevation(Transform):
    """Compute pointwise elevation with respect to the ground.

    We do so in a two-step process where we first remove as many
    potentially non-ground points as possible, before fitting a surface
    model to the resulting trimmed cloud to estimate the ground surface.

    We offer several tools for filtering out non-ground points:
    - filtering out all points that are higher than a given threshold
      above the lowest point in the cloud
    - filtering out all points whose local verticality (see
      `PointFeatures`) is above a given threshold
    - projecting points into a horizontal XY grid and only keeping the
      lowest point of each XY bin

    We offer several tool for modeling the ground surface from the
    trimmed cloud:
    - 'ransac': single planar surface using RANSAC
    - 'knn': linear interpolation of the k nearest trimmed points
    - 'mlp': piecewise planar approximation with an MLP

    :param z_threshold: float
        Ground points will be first searched within `global_threshold`
        of the lowest point in the cloud. Adjust this if the lowest
        point is below the ground or if you have large above-ground
        planar structures
    :param verticality_threshold: float
        Ground points will be searched among those with lower
        verticality than `verticality_threshold`. This assumes
        verticality has been computed beforehand using `PointFeatures`.
        Note that, depending on the chosen value, this will also remove
        steep slopes
    :param xy_grid: float
        Bin all points into a regular XY grid of size `xy_grid` and
        isolate as candidate ground point for each cell the one with
        the lowest Z value
    :param model: str
        Model used for fitting the ground surface. Supports: 'ransac',
        'knn', and 'mlp'.
        'ransac':
         Model the ground as a single plane using RANSAC.
        'knn':
        Model the ground based on a trimmed point cloud carrying ground
        points only. At inference, a point is associated with its
        nearest neighbors in L2 XY distance in the reference ground
        cloud. The ground surface is estimated as a linear interpolation
        of the neighboring reference points. The elevation is then
        computed as the corresponding gap in Z-coordinates.
        'mlp':
        Fit an MLP to a point cloud. Assuming the point cloud mostly
        contains ground points, this function will train an MLP to model
        the ground surface as a piecewise-planar function.
    :param scale: float
        Scaling by which the computed elevation should be divided, for
        the sake of normalization. NB: passing `scale <= 0` will
        skip this transform altogether. This can typically be used for
        bypassing it in the configs
    :param kwargs: dict
        Arguments that will be passed down to the surface modeling
        function
    """

    def __init__(
            self,
            z_threshold=None,
            verticality_threshold=None,
            xy_grid=None,
            model='ransac',
            scale=3.0,
            **kwargs):
        if verticality_threshold is not None:
            assert 0 < verticality_threshold < 1
        if xy_grid is not None:
            assert xy_grid > 0
        assert model in ['ransac', 'knn', 'mlp']

        self.z_threshold = z_threshold
        self.verticality_threshold = verticality_threshold
        self.xy_grid = xy_grid
        self.model = model
        self.scale = scale
        self.kwargs = kwargs

    def _process(self, data):
        # Skip elevation computation
        if self.scale <= 0:
            return data

        # Recover the point positions
        pos = data.pos

        # Initialize a mask for the filtering out as many non-ground
        # points as possible, to facilitate the subsequent search of the
        # ground surface in the point cloud
        mask = torch.ones(data.num_points, device=pos.device, dtype=torch.bool)

        # See `filter_by_z_distance_of_global_min` for more details
        if self.z_threshold is not None:
            mask = mask & filter_by_z_distance_of_global_min(
                pos, self.z_threshold)

        # See `filter_by_verticality` for more details
        if self.verticality_threshold and (0 < self.verticality_threshold < 1):
            if not hasattr(data, 'verticality'):
                raise ValueError(
                    f"The Data object does not have a 'verticality' attribute. "
                    f"To compute verticality, please call PointFeatures on "
                    f"your Data first")
            mask = mask & filter_by_verticality(
                data.verticality, self.verticality_threshold)

        # See `filter_by_local_z_min` for more details
        if self.xy_grid:
            mask = mask & filter_by_local_z_min(pos, self.xy_grid)

        # Trim the point cloud based on the computed filters. We hope
        # that there are mostly ground points in there, but can't be
        # 100% sure
        pos_trimmed = pos[mask]

        # Fit a model to the trimmed points
        if self.model == 'ransac':
            model = single_plane_model(
                pos_trimmed, 
                **filter_kwargs(single_plane_model, self.kwargs))
        elif self.model == 'knn':
            model = neighbor_interpolation_model(
                pos_trimmed, 
                **filter_kwargs(neighbor_interpolation_model, self.kwargs))
        elif self.model == 'mlp':
            model = mlp_model(
                pos_trimmed, 
                **filter_kwargs(mlp_model, self.kwargs))

        # Compute the elevation of each point wrt the estimated ground
        # surface
        elevation = model(pos).view(-1, 1)

        # Scale the elevation and save it in the Data object
        data.elevation = elevation.view(-1, 1) / self.scale

        return data


class RoomPosition(Transform):
    """Compute the pointwise normalized room coordinates, as defined
    in the S3DIS paper section 3.2:
    https://openaccess.thecvf.com/content_cvpr_2016/papers/Armeni_3D_Semantic_Parsing_CVPR_2016_paper.pdf

    Results will be saved in the `pos_room` attribute.

    NB: this is rather suited for indoor setting, with regular
    dimensions and not so much for open outdoor clouds with unbounded
    sizes.

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

        level_p = fill_list_with_string_indexing(
            level=self.level,
            default=-1,
            value=self.p,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        transforms = [
            ColorAutoContrast(p=p, blend=self.blend, x_idx=self.x_idx)
            for p in level_p]

        nag.apply_data_transform(transforms)

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

        level_p = fill_list_with_string_indexing(
            level=self.level,
            default=-1,
            value=self.p,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        transforms = [ColorDrop(p=p, x_idx=self.x_idx) for p in level_p]

        nag.apply_data_transform(transforms)

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

        level_mean = fill_list_with_string_indexing(
            level=self.level,
            default=[0, 0, 0],
            value=self.mean,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        level_std = fill_list_with_string_indexing(
            level=self.level,
            default=[1, 1, 1],
            value=self.std,
            output_length=nag.absolute_num_levels,
            start_index=nag.start_i_level)

        transforms = [
            ColorNormalize(mean=mean, std=std, x_idx=self.x_idx)
            for mean, std in zip(level_mean, level_std)]

        nag.apply_data_transform(transforms)

        return nag

class PretrainedCNN(Transform):
    """
    Forwards the data through a module that has been already been trained.
    This is useful to call during preprocessing in order to get the point
    features on which the partition is computed.
    
    :param first_stage: src.nn.stage.PointStage
        The lightweight module embedding the points.
        Typically, a sparse CNN of 3 layers.

    :param ckpt_path: str
        The path to the checkpoint, to be loaded in the `first_stage` 
        module.
        
    :param norm_mode: str
        Indexing mode used for feature normalization. This will be
        passed to `Data.norm_index()`. 'graph' will normalize
        features per graph (i.e. per cloud, i.e. per batch item).
        'node' will normalize per node (i.e. per point). 'segment'
        will normalize per segment (i.e.  per cluster)
        NB : same as in `SPT`
    
    :param device: str
        The device on which to load the first stage module.
        
    """
    def __init__(
        self, 
        first_stage, 
        ckpt_path, 
        partition_hf=None, 
        norm_mode='graph', 
        device='cuda',
        verbose=False):
        
        self.first_stage = first_stage
        self.norm_mode = norm_mode
        self.ckpt_path = ckpt_path
        self.partition_hf = partition_hf
        self.device = device
        self.verbose = verbose
        
        assert self.partition_hf is not None, \
            "`partition_hf` has not been set up."

        self.first_stage.to(self.device)
        log.info(
            f"Initializing {self.__class__.__name__} (transform used in "
            f"preprocessing to compute point features for the partition) with: "
            f"{self.ckpt_path}")
        
        self.first_stage = self.load_checkpoint(
            self.first_stage,
            self.ckpt_path,
            self.device,
            verbose=self.verbose)
    
    def _process(self, data):
        
        with torch.no_grad():
            # In this transform, we don't delete the partition_hf features after adding them to 'x'
            # as we need to save them on disk for the training procedure.
            data.add_keys_to(keys=self.partition_hf, to='x', delete_after=False)
            
            x, diameter = SPT.forward_first_stage(
                data=data,
                first_stage=self.first_stage,
                use_node_hf=True,
                norm_mode=self.norm_mode,
            )

            data.x = x
        
        return data

    @staticmethod
    def load_checkpoint(
            first_stage,
            ckpt_path,
            device,
            verbose=False):
        """
        Load the checkpoint and update the first stage module.
        """
        assert ckpt_path is not None and os.path.exists(ckpt_path), \
            f"Expected a valid checkpoint path. Received {ckpt_path=}"

        checkpoint = torch.load(ckpt_path, map_location=device)
        model_dict = first_stage.state_dict()
        
        # Filtering the key
        first_stage_keys = [
            k for k in checkpoint['state_dict'].keys()
            if 'first_stage' in k]
        ckpt_dict = {
            k.replace('net.first_stage.',""): checkpoint['state_dict'][k]
            for k in first_stage_keys}
        
        loadable_params = {}
        unused_params = {}
        
        for k,v in ckpt_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    loadable_params[k] = v
                else:
                    raise ValueError(
                        f"Shape mismatch for {k}: checkpoint {v.shape} vs "
                        f"model {model_dict[k].shape}")
            else:
                unused_params[k] = v

        missing_params = {
            k: v
            for k, v in model_dict.items()
            if k not in loadable_params}
        
        # Load the parameters
        model_dict.update(loadable_params)
        first_stage.load_state_dict(model_dict)
        
        if verbose:
            log.info(
                f"Loaded {len(loadable_params)} parameters from "
                f"checkpoint: " + ', '.join(loadable_params.keys()))
            log.info(
                f"Unused {len(unused_params)} parameters from the "
                f"checkpoint (not in model): " +
                ', '.join(unused_params.keys()))
            log.info(
                f"Missing {len(missing_params)} model parameters "
                f"(not in checkpoint, keeping initialization): " +
                ', '.join(missing_params.keys()))
            
        return first_stage
