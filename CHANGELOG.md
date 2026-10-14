# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## \[3.X.X\] - XXXX-XX-XX

### Added

- Hybrid `Muon` optimizer (`src/optim/muon.py`), Muon on the 2D hidden weights
and AdamW on the others, enabled with `+optimizer=muon`. The default optimizer
is unchanged; with lr 0.01, EZ-SP gains +2.3 mIoU on KITTI-360 and +1.8 on DALES

- `experiment=semantic/{kitti360,dales}_ezsp_cpu_sampling`: EZ-SP with the
subgraph sampling in `train_transform`, run by the DataLoader workers, thanks
to the new `as_list` parameter of the subgraph sampling transforms (EZ-SP train
step -11% to -28%, [#245](https://github.com/drprojects/superpoint_transformer/pull/245))
- `DataLoader(cross_epoch_prefetch=True)`, exposed as
`datamodule.dataloader.cross_epoch_prefetch` (default `True` in
`configs/datamodule/semantic/default.yaml`, used by the train and val
dataloaders): the workers keep prefetching across epoch boundaries. PyTorch
workers only start loading an epoch when `iter(loader)` is called, even with
`persistent_workers=True`, so each epoch started by waiting for its first
batches (about 1 s per epoch for EZ-SP DALES on an RTX A6000). The shuffling
still changes at each epoch, but a given seed no longer gives the same order as
before. Set it to `False` to recover the previous behavior

### Changed

- Moved the graph connected components and graph partitioning algorithms 
introduced in EZ-SP to a standalone library: 
[torch-graph-components](https://github.com/drprojects/torch-graph-components). 
This dependency is now installed by `install.sh`.

- Partition training update:
  - Logging of the number of inter-edges per epoch the model is trained on
    (`train/n_inter_edge`)
  - Clearer error or warning messages. Distinct messages for:
    - error: no edge at all in the batch;
    - warning: only void edges in the batch;
    - warning: only intra-edges and no inter-edges in the batch.
  - The two latter cases no longer raise an error: if a batch is composed of a
    random crop in an area with only one semantic label, training continues.
    However, if no inter-edge is found during a whole training epoch, an error
    is raised, as this likely indicates a bug in the graph or crop definition.

- **⚠️ Breaking Change**: the code version is now `3.1.0`. Model checkpoints
(SPT, EZ-SP, SPC) saved with an older version (`metadata['__version__']` below
`3.1.0`, or missing) now raise an error when loaded. The fixes of this version
change the features and graphs the models see, so older weights are not
compatible with this code: **please retrain**, or use the code release matching
your checkpoint. For development only, setting `allow_old_checkpoint=true` (in
`train.yaml` / `eval.yaml`, or as a `load_from_checkpoint()` argument) loads
them anyway, with a warning. The EZ-SP partition CNN checkpoints loaded by
`PretrainedCNN` are not checked: the CNN inputs are not affected by these fixes

- `BaseDataset` caches `cloud_ids`, `processed_file_names` and
`processed_paths`, which were rebuilt over the whole dataset at every
`__getitem__`
- `SelfAttentionBlock` computes its linear edge RPEs (keys, queries, values)
with one matmul (EZ-SP train step -4%, [#244](https://github.com/drprojects/superpoint_transformer/pull/244))
- Faster and more precise `GraphNorm`: per-graph sums as one-hot matrix
products instead of a scatter (EZ-SP train step -8%, [#240](https://github.com/drprojects/superpoint_transformer/pull/240))
- `Data.select()` computes `num_nodes`, `num_edges` and the edge keys once
instead of once per attribute ([#239](https://github.com/drprojects/superpoint_transformer/pull/239))
- Semantic configs (`instance: False`) no longer load the instance labels `obj`
(EZ-SP train step -11%, [#238](https://github.com/drprojects/superpoint_transformer/pull/238))
- Faster attention: query-key products computed without `torch.einsum`
(EZ-SP train step -21%, [#237](https://github.com/drprojects/superpoint_transformer/pull/237))
- Faster preprocessing with `xy_tiling` or `pc_tiling`: each raw cloud is now
read once for all its tiles, instead of once per tile (eg 9 times per DALES
tile with `xy_tiling: 3`). `BaseDataset.process()` keeps the last raw cloud
read in memory and reuses it for the following tiles of the same raw cloud.
On 3 DALES tiles (27 sub-tiles, without `pre_transform`), reading, tiling and
saving go from 59.1 to 13.9 s. This only saves reads if the tiles of a raw
cloud are processed consecutively, as with the default `processed_paths`
order: the number of reads is logged at the end of the preprocessing, with a
warning if a raw cloud was read several times. Preprocessing outputs and hash
are unchanged
- Removed some avoidable CPU-GPU syncs in the model: the default
qk-scale of the attention and pooling blocks counts the neighbors with a
known output size instead of `bincount` (`build_qk_scale_func` functions
now accept an optional `num_nodes`), and `scatter_mean_weighted` /
`UnitSphereNorm` no longer read GPU values on CPU (the non-negative
weights check of `scatter_mean_weighted` now only runs in debug mode).
Outputs are unchanged. The speedup is negligible for the published
configs, whose train step is GPU-bound.
- Removed avoidable CPU-GPU syncs from the on-device transforms (~1300 -> 176
per EZ-SP training batch, train step -8%, [#233](https://github.com/drprojects/superpoint_transformer/pull/233)).
`RandomAxisFlip`, `ColorAutoContrast` and `ColorDrop` now draw on the CPU
generator (different draws for a given seed)

### Deprecated

### Fixed

- `MultiLoss.state_dict()` raised a `KeyError` before the class weights were
set, and `MultiLoss.load_state_dict()` failed on its own `state_dict()`
([#247](https://github.com/drprojects/superpoint_transformer/issues/247))
- `fast_randperm` ignored the seed on CPU (numba RNG), so CPU sampling was not
reproducible; it now uses `torch.randperm`
([#248](https://github.com/drprojects/superpoint_transformer/issues/248))
- `OnTheFlyHorizontalEdgeFeatures`: at train time, about half of the
horizontal edges were flipped without flipping the sign of their `mean_off`
edge feature, which then pointed the wrong way. See
[#232](https://github.com/drprojects/superpoint_transformer/pull/232) for
details
- `WeightedFocalLoss` no longer reduces the loss twice:
  - with `reduction='mean'`, it returned `mean(focal_term) * mean(CE)`
  instead of `mean(focal_term * CE)`, so the focal term did not modulate each
  sample's loss;
  - with `reduction='none'` (used by `loss_with_target_histogram`, eg for
  levels >= 2 with `loss_type='ce_kl'`), the loss was divided by the number of
  items, which nearly disabled the supervision of the upper partition levels.

  With `gamma=0`, both paths now match `nn.CrossEntropyLoss`. This affects
  models trained with `WeightedFocalLoss`, such as the EZ-SP DALES and
  KITTI-360 configs
- `loss_type='wce'` and `loss_type='wce_kl'`: the dominant-label histogram
was written into every row instead of only the row of each node (eg
`[[5, 1, 0], [0, 2, 7]]` gave `[[6, 0, 9], [6, 0, 9]]` instead of
`[[6, 0, 0], [0, 0, 9]]`), also in the multi-stage `wce` loss
([#242](https://github.com/drprojects/superpoint_transformer/pull/242))
- **⚠️ Breaking Change**: `cluster_radius_nn_graph`, which builds the
horizontal graph of each partition level in `RadiusHorizontalGraph` (and the
instance graph of `OnTheFlyInstanceGraph` for SuperCluster), compared squared
distances (as returned by `knn_1`) to a distance threshold. All pairs of superpoints whose
bounding-box centers were further apart than about `sqrt(r_max_edge)` were
discarded, whatever `gap`, making the graphs sparser than `graph_gap` implies,
mostly at the upper levels where superpoints are large. Mean degree at
levels 1 / 2 / 3, measured on two tiles of each dataset:
  - DALES (`graph_gap: [5, 30, 30]`): from 26-28 / 29 / 8 to 33 / 34 / 34,
  with a preprocessing time increased by about 10%;
  - KITTI-360 (`graph_gap: [3, 10, 30]`): from 33 / 32 / 22-26 to
  34 / 35 / 35, with an unchanged preprocessing time

  This changes the preprocessing outputs without changing its parameters,
  hence without changing the dataset hash: **you will need to delete your
  already-existing preprocessed datasets and re-run the preprocessing**,
  otherwise the former graphs will be silently reused
- **⚠️ Breaking Change**: `use_diameter=True` only injected a column of zeros:
`DownNFuseStage` and `UpNFuseStage` did not forward the `diameter` they
received to `Stage.forward()`, and the up stages were passed `diameter=None`.
The node diameter is now injected, as `log(1 + diameter)`. Since the EZ-SP
configs (`experiment/semantic/default_ezsp.yaml`) set `use_diameter: True`,
their models were in fact trained with a constant zero feature. These configs
now use `use_diameter: False`, the default of all other configs. Models
trained with `use_diameter: True` before this version are not compatible with
the fixed behavior
- `NAG.select()`: when the selection kept all the nodes of a level, in the
same order (eg dropping a few level-1 nodes while keeping all their level-2
parents), the levels above that one lost their `sub`, and symmetrically the
levels below lost their `super_index`. A later selection on these levels then
broke the hierarchy, or crashed when instance labels (`obj`) were loaded. This
can happen in the on-device train transforms (`SampleRadiusSubgraphs`, then
`SampleSegments`), eg when the sampled subgraph covers a whole small cloud
- Degenerate clouds and graphs no longer crash preprocessing, training or
inference. Clouds where a partition level has a single superpoint or no
possible edge (eg tiny, sparse or collinear clouds) now go through the whole
pipeline and keep the expected number of partition levels:
  - `CutPursuitPartition`: a level with a single node or without edges gives
  an identity level instead of stopping the hierarchy. A level with exactly 2
  edges no longer crashes the conversion to forward-star, which also affected
  the SuperCluster instance partition;
  - `RadiusHorizontalGraph`, `OnTheFlyHorizontalEdgeFeatures` and `SPT`
  support levels without horizontal edges;
  - `Data.connect_isolated()` supports a single node, coincident nodes, and
  less than `k` other nodes (which previously created edges to `-1`);
  - `Data.select()` keeps an empty `edge_index` consistent with `edge_attr`,
  and `Data.to_trimmed()` accepts graphs without edges;
  - `GroundElevation` with `model='ransac'` falls back to the horizontal plane
  through the lowest point when no plane can be fitted (eg less than 3 ground
  candidates);
  - `scatter_pca` retries on CPU when the GPU eigendecomposition fails to
  converge on ill-conditioned covariances (eg collinear points);
  - `SampleRadiusSubgraphs` no longer samples more seeds than a batch item has
  nodes;
  - the semantic and panoptic losses of a batch without any labeled point or
  supervised edge are null instead of NaN.

  These code paths are only reached by such degenerate inputs: the
  preprocessing hash and the outputs on regular clouds are unchanged
- `GridSampling3D` with `chunk_size`: the `sub` Cluster indices of clouds
larger than `chunk_size` were offset when merging the chunks, pointing out of
the input cloud (eg max index 793 for a 200-point cloud). This broke the
mapping from level-0 voxels to full-resolution points, eg in
`full_res_semantic_pred()`. Preprocessed datasets with clouds larger than
`chunk_size` (10M points by default, eg S3DIS areas) must be preprocessed
again to benefit from this fix, the preprocessing hash being unchanged.
The same chunk merging also renumbered `super_index` and offset `obj_pred`
(`obj` was already handled), and dropped `batch`. These attributes are not
present when voxelizing raw clouds in the default pipelines, but are now kept
unchanged as well
- `OnTheFlyHorizontalEdgeFeatures`: `angle_source` and `angle_target` are now
swapped for the j->i edges appended to the trimmed i->j graph. They used to be
copied unchanged, so for half of the edges, `angle_source` described the
normal of the target node and `angle_target` that of the source node
- `instantiate_datamodule_transforms()` silently skipped a transform that
could not be instantiated, only logging the error. A datamodule could then
eg run the preprocessing without its `pre_transform`, or without one of its
augmentations. The error is now raised
- `RandomAnisotropicScale` no longer inflates the `std_off` and `mean_dist`
edge features by ~sqrt(3) at train time. They were multiplied by
`scale.norm()` (~sqrt(3) for a ~(1, 1, 1) scale), causing a train/eval shift
and pushing `std_off` beyond its clip value of 2. They are now scaled by the
isotropic equivalent of the scale, `scale.norm() / sqrt(3)`, with a square
root for `mean_dist`, and `std_off` is clipped like at preprocessing time.
The random scale is also drawn directly on the device, which removes a CPU-GPU
synchronization at each call
- `RandomAnisotropicScale` was isotropic, unlike documented: a single random
draw was shared by the 3 axes, so that the scaling was isotropic with the
scalar `delta` of all configs. A scaling factor can now be drawn for each axis,
with the new `per_axis` parameter:
  - `RandomAnisotropicScale(per_axis=True)`, the default of the class, is
  anisotropic, as documented;
  - the datamodule configs set it with `anisotropic_scaling_per_axis`, which is
  `False` for now: the configs keep the isotropic scaling the published models
  were trained with, until the anisotropic one is evaluated.

  Normals are now divided by the scale instead of multiplied by it, which only
  made a difference for anisotropic scalings. With an anisotropic scaling, the
  `std_off` and `mean_dist` edge features are approximated with the isotropic
  equivalent of the scale
- `test/loss` was logged twice by `SemanticSegmentationModule`
- `RandomTiltAndRotate` skipped the rotation when `phi=0`: KITTI-360 was never
rotated, its config now sets `tilt_n_rotate_theta: 0` to keep this behavior
([#241](https://github.com/drprojects/superpoint_transformer/pull/241))
- `RandomTiltAndRotate` drew the tilt with a standard deviation `sqrt(sigma)`
instead of `sigma`: DALES, S3DIS and ScanNet now use `tilt_n_rotate_phi: 4.15`
to keep the same tilts ([#241](https://github.com/drprojects/superpoint_transformer/pull/241))

### Removed

## \[3.0.0\] - 2025-11-27

### Added

- Introduced a `TensorHolderMixIn` class from which `CSRData` and `NAG` inherit
for code factorization and easier future maintenance. This mixin mimics 
mechanisms of torch-geometric's `Data`, in particular the use of `items()` and
`apply()` for easily implementing important torch-like operations such as
`clone()`, `to()`, `cpu()`, `cuda()`, `detach()`, `pin_memory()`...
- Introduced `to_flat_tensor()` and `from_flat_tensor()` mechanisms for 
converting our data structures to dictionaries of flat tensors, organized by 
dtypes
- Introduced `print_memory_summary()` and `nbytes` for our data structures to 
facilitate inspecting the breakdown of memory usage of each object
- Added `load_non_fp_to_long` attribute for `BaseDataset`. Non-floating-point 
tensors are saved with the smallest precision-preserving dtype possible. 
`load_non_fp_to_long` rules whether these should be cast back to int64 upon 
reading. To save memory and I/O time, we recommend setting 
`load_non_fp_to_long=False` and using the `Cast` or `NAGCast` Transform in 
your `on_device_transform`. This allows postponing the casting to GPU and 
accelerates reading from disk and CPU-GPU transfer for the DataLoader 
- `SampleRadiusSubgraphs(cylindrical=True)` now allows for cylindrical sampling,
rather than spherical sampling. This can be useful for outdoor environment and
is set as the default behavior in the configs for DALES and KITTI-360
- Added `local/` directory to the `.gitignore` to place local scripts and 
experiments you might not want to track
- Improved `PointFeatures` and `SegmentFeatures` to support GPU-based 
computation, which is much faster than the CPU-based `pgeof` features, while
avoiding device transfers when inputs are on GPU. The features of the GPU and 
CPU versions are the same, up to a certain precision, due to the stochasticity
of some GPU operations. This is sufficient for our needs in this project
- Improved `GroundElevation` to support GPU-based computation for 
`mode='ransac'`. Uses the `torch-ransac3d` library, which is much faster than 
`sklearn`'s CPU-based `RANSACRegressor`, while avoiding device transfers when
inputs are on GPU 
- Introduced `neighbors_dense_to_csr` to convert a `[N, K]` tensor of dense 
neighbor indices to CSR format
- Introduced a 'chunking' (i.e. sharding) mechanism in `GridSampling3D` to avoid
CUDA OOM when voxelizing very large clouds (e.g. S3DIS Area 5)
- Introduced a `forget_batching` function to convert `NAGBatch`, `Batch`, 
`CSRBatch` to `NAG`, `Data`, `CSRData`, respectively
- All `Transform` objects can now pass `__call__(..., verbose=True)` to print
their metainformation and processing time. This can be typically used to 
benchmark preprocessing times. NB: this internally triggers a CPU-GPU sync so 
make sure to have `verbose=False` when you are actually training or inferring
- Introduced the option `BaseDataset.process(verbose=True)`, which triggers 
printing per-cloud preprocessing times. This behavior is not exposed to the rest
of the framework, it is more intended for debugging purposes at the moment. This
can also be triggered in debug mode with `src.debug()`
- Added hydra parameter `model.weighted_loss_smooth` in 
`model/semantic/default.yaml`
- Added hydra parameter `model.point_out_dim` in `model/semantic/default.yaml`
to control more easily the output dimension of the First Stage (especially when
using a CNN, cf. override of `point_out_dim` in `_point_cnn.yaml`). Therefore, 
this new parameter is used in the following config files to properly define the
needed dimensions: `_down.yaml`, `spt-2` and `spt-3`
- Added hydra parameter `model._point_injection_dim`
- Added `spt-0` and `spt-1`, for models processing Data with hierarchical 
partition of respectively 0 and 1 level
- Added config `debug/pytorch_profiler.yaml` to use the PyTorchProfiler 
(e.g., `python src/train.py experiment=semantic/s3dis debug=pytorch_profiler`)
- Added 3 parameters to `Cast`
  - `pos_dtype`: optional parameter to determine the casting dtype of pos. 
  Independently of `pos_dtype`, high precision is preserved thanks to the 
  `pos_offset` parameter, whose dtype remains unchanged.
  - `rgb_to_float`: whether to cast RGB to float. If False, byte.
  - `optimal_int_dtype`: controls integer casting - whether to automatically 
  find the smallest integer dtype that can represent the tensor without overflow.
- Added `train_on_val` attribute to `BaseDataModule` to replace the trainset by
the validation. Useful for debugging purposes.
- Added `store_features` attribute to `SPT`: to toggle the saving of the 
features processed by the network and the input features. Useful for 
visualization purposes
- Introduced `add_keys_to` method to `Data` and `NAG` classes
- `NAGDropoutColumns`, `NAGDropoutRows` and `NAGJitterKey` now supports a list 
of keys  (previously, only a single key was supported)
- Added `outputs/`, `wandb/` and `ckpt/` in `.gitignore`
- Added `prepare_only_test` attribute. It allows you to run `eval.py` without 
needing to preprocess the train and val datasets. Note that we don't log the 
computed loss, as the loss is weighted by the training set statistics which are 
not available.

- Sparse convolutions in first stage:
  - Introduced a `SparseCNN` class to compute sparse convolutions thanks to the
  [torchsparse library](https://github.com/mit-han-lab/torchsparse)
  - Added hydra parameter `model.net.point_cnn_blocks=False` in `_point.yaml`
  - Added `cnn_blocks` boolean attribute to `PointStage` and `SPT`. It controls 
  the initialization and use of a `SparseCNN` at the beginning of the 
  `PointStage` / `first_stage`.
  - Added `cnn`, `cnn_kernel_size`, `cnn_dilation`, `cnn_norm`, 
  `cnn_activation`, `cnn_residual`, `cnn_global_residual` to parametrize the 
  CNN.
  - Added `point_mlp_on_cnn_feats` attribute to use the CNN features as input 
  to the point MLP; otherwise the MLP processes handcrafted features and the 
  CNN and the MLP are concatenated.
  - Added new hydra parameter `datamodule.post_cnn_point_hf` in
  `_features.yaml` (parameter relevant only when a CNN is used in the first 
  stage)
  - Added new hydra parameter `datamodule.all_point_hf` in `_features.yaml` to
  summarize the point features to be loaded for training
  - Added `SemanticSegmentationModule.initialize_cnn` method that controls :
    - whether the cnn is trained from scratch or using the checkpoint from
    `model.pretrained_cnn_ckpt_path` hydra parameter,
    - whether the cnn is frozen or not.

- EZ-SP | partition training:
  - Introduced `PartitionAndSemanticModule` that inherits from
  `SemanticSegmentationModule`. Depending on its boolean parameter
  `training_partition_stage`, it learns features for the partition algorithm or
  train the semantic (on the partition that has been trained).
  - Introduced `PartitionOutput` class to store the output of the model during
  partition training (in the same spirit of `SemanticSegmentationOutput` and
  `PanopticSegmentationOutput`)
  - Added the proper initialization of `Conv3d` layers in
  `src/utils/nn.py/init_weights`
  - Introduced `BinaryFocalLoss`, a simpler version `WeightedFocalLoss` for 
  binary target. This loss is used when training the features for the 
  partitioning algorithm on the proxy task intra/inter edge classification
  - Introduced `PartitionCriterion` class that gathers useful function around
  the partition learning objective, notably mapping features to edge affinities,
  doing adaptive sampling to balance between inter and intra edges.
  - Introduced util function `compute_edge_distances_batch`, used to map 
  features to edge affinities (src/utils/batch_utils.py) in `PartitionCriterion`

- EZ-SP | semantic training
  - Introduced `QuantizePointCoordinates` transform to quantize the position
  attribute `pos` in coordinates `coords`. These are needed for the sparse
  convolutions.
  - Added `QuantizePointCoordinates` at the end of the on-device-transforms in
  `datamodule/semantic/<dataset>.yaml`, which computes the quantized coordinates
  required by the sparse CNN (if used)
  - Added `datamodule.quantize_coords` hydra parameter 
  - Introduce `PretrainedCNN` transform that forwards the data through a CNN 
  that has already been trained. Useful to call during preprocessing in order to
  get the point features on which the partition is computed.
  - Added Damien's Algo (`components_merge.py` & `connected_components.py`). To
  be moved later to a dedicated isolated repo.
  - Introduced `GreedyContourPriorPartition`, which leverages
  `merge_components_by_contour_prior` function to compute a hierarchical
  partition.

- EZ-SP | Configs
  - Added config files for EZ-SP (`configs/`):
    1. to learn the partition
      - datamodule configs:
        - `datamodule/partition/default_ezsp.yaml`
        - `datamodule/partition/s3dis_ezsp.yaml`
        - `datamodule/partition/kitti360_ezsp.yaml`
        - `datamodule/partition/dales_ezsp.yaml`

      - experiment configs:
        - `experiment/partition/s3dis_ezsp.yaml`
        - `experiment/partition/kitti360_ezsp.yaml`
        - `experiment/partition/dales_ezsp.yaml`

    2. to learn the semantic
    - datamodule configs:
      - `datamodule/semantic/default_ezsp.yaml`
      - `datamodule/semantic/s3dis_ezsp.yaml`
      - `datamodule/semantic/kitti360_ezsp.yaml`
      - `datamodule/semantic/dales_ezsp.yaml`
    - experiment configs:
      - `experiment/semantic/default_ezsp.yaml`
      - `experiment/semantic/s3dis_ezsp.yaml`
      - `experiment/semantic/kitti360_ezsp.yaml`
      - `experiment/semantic/dales_ezsp.yaml`

- Introduced a new [`VersionHolder`](src/utils/version.py) class to manage
code version across the model architecture. A `VersionHolder` is now created
in `SPT.__init__` and shared by all stages (`Stage`, `PointStage`,
`DownNFuseStage`, `UpNFuseStage`) to ensure consistent version handling
throughout the network. The version can be updated via the setter property
`spt.version = ...`, which automatically propagates to all stages the update.
This is implemented to preserve compatibility with the official weights of SPT 
and SPC released before the [commit a0f753b](https://github.com/drprojects/superpoint_transformer/commit/a0f753b35b86e06d426113bdeac9b0123b220aa3) fixing the residual 
connection. If you have trained checkpoints after this commit, please use the
migration script [`add_version_to_checkpoint.py`](src/utils/backwards_compatibility/add_version_to_checkpoint.py)
to set version `"2.2.0"` to your checkpoint's metadata.
`__version__` is defined in [src/\_\_init\_\_.py](src.__init__.py).

### Changed

- **⚠️ Breaking Change**: modified (again 🙈) the serialization behavior of the 
data structures. You will need to re-run all your already-existing datasets' 
preprocessing to adjust to the new serialization format. It would also be 
possible to write a script that would adjust the existing HDF5 files, but we do 
not provide such script
  - Introduced [`convert_nag_v2_to_v3.py`](src/utils/backwards_compatibility/convert_nag_v2_to_v3.py)
  to convert saved NAG files from version V2 of the SPT's codebase to V3 (current
  version). **Note**: The script does not handle `NAGBatch`. (If attempting to 
  convert a `NAGBatch`, it will be converted as a `NAG` - i.e., the keys specific
  to the batch are removed).
  Usage: `python -m src.utils.backwards_compatibility.convert_nag_v2_to_v3 nag_v2.h5`
- Moved `NAGCast` as the first on-device transform for all datasets by default. 
This prioritizes the casting of tensors saved with a smaller dtype to save 
memory and dataloading time. On the longer run, we may move this back to a 
subsequent step in the on-device transforms, as long as the preceding transforms
are not dtype-sensitive (e.g. indexing with a non-long tensor, overflowing index
tensors by trying to increase some indices, ...)
- Nano pipeline no-longer needs to load atom-level Data, saving memory and 
compute. (Atoms are currently limited to points, and pixels in the future.)
- **⚠️ Breaking change**: `NAG.__getitem__` returns a Data object from the 
hierarchy, according to the **absolute** level index, while only **relative** 
index was handled so far. 
  - Concretely, **level-0 now always refers to the atomic level**. To mimic 
  the older `NAG.__getitem__` behavior (indexing relative to the loaded levels
  of the `NAG`object), use `NAG._list[relative_idx]` (i.e. : `NAG._list[0]`
  returns the first loaded level). 
  - The last absolute level of a `NAG` is the last loaded level in the `NAG`.
  (It means that the level `NAG.end_i_level` is loaded by definition.)
- `NAG` objects hold a new attribute `start_i_level` (new keyword argument 
be passed to `NAG.__init__(data_list, start_i_level = 0)`), indicating the absolute
index of the first `Data` level loaded in the object.
- new `NAG` properties to handle the hierarchy (`end_i_level`, `has_atoms`, 
`first_level`, `absolute_num_levels`, `level_range`)
- **⚠️ Breaking Change**: renamed the `adjacency_mode='radius'` of 
`OnTheFlyInstanceGraph` into the more explicit `adjacency_mode='radius-atomic'`
- The behavior of `tensor_idx()`, as well as all `.select()` and `.load()` 
functions was a bit modified. Passing `idx=None` will still behave as if no 
indexing was required. However, the behavior changed for passing an empty 
indexing like `idx=[]`, or `idx=np.array([])`, or `idx=torch.tensor([])`, or 
`idx=slice(i, i)`. Empty indexing will now result in empty objects, instead of
ignoring the indexing as we were doing previously.
- Overwrote pytorch lightning's `LightningDataModule.transfer_batch_to_device()`
to implement our own device transfer. The behavior is the same as before, but
allows explicitly playing with how `transfer_batch_to_device` is done for our 
custom data structures (`Data`, `NAG`). In particular, this allows exposing the
`non_blocking` option, which can now be triggered from the configs with the 
`datamoule.non_blocking` parameter.
- `SampleKHopSubgraphs(hops=...)` with `hops < 0` now skips the transform, 
similar to `SampleRadiusSubgraphs(r=...)` with `r <= 0`
- if `feats` given to `feats_to_plotly_rgb` are 3-dimensional but have negative values,
then it now rescales it to $[0,1]^3$ (affine transformation - the same after PCA when feats
have more than 3 dimension)
- `Transform` object can now have a Data as input even if `_IN_TYPE = NAG`. Under the hood, the transform wraps the Data object in a NAG, processes it, then unwraps. 
- `GridSampling3D` coordinates quantization now maps coordinates to only positive values by default to optimize torchsparse performance. This can be revoked with the newly introduced parameter `allow_negative_coords=True`.
- `GroundElevation` can now be skipped if `scale=0`
- `KNN` can now be skipped with `r_max<=0` or `k<=0`
- Preprocessing can now save `Data` objects and on-device-transforms can now handle end-to-end `Data` objects. This required the following changes:
  - `NAG.load` called on a file storing a `Data` object no longer raises an error and instead returns the appropriate Data object
  - `BaseDataset.__init__` for proper Data loading for in memory datasets
  - `BaseDataset._process_single_cloud` : call `Data.save` instead of `NAG.save` if needed
  - `BaseDataset.__getitem__` : call `Data.load` instead of `NAG.load` if needed
  - `BaseDataset.get_class_weight` : call `Data.load` instead of `NAG.load` if needed
  - `BaseDataModule.on_after_batch_transfer` can now handle `Data` objects (in addition to `NAG`s)
- Factorization of `on_validation_epoch_end` and `on_test_epoch_end` in `_on_eval_epoch_end` of `SemanticSegmentationModule`.
- Factorization of `on_train_epoch_end` in `_on_train_epoch_end` of `SemanticSegmentationModule`
- `coords` is now part of `point_no_save_keys` (cf. `_features.yaml`)
- `brute_force=True` in `SampleRadiusSubgraphs` transform in all `datamodule/semantic/<dataset>.yaml` (`<dataset>`: `s3dis`, `dales`, `kitti360` and `scannet`). This was changed to avoid accelerate the transform.
- Factorization of the pcp related parameters of `datamodule/semantic/<dataset>.yaml` configs in the new config files `datamodule/semantic/<dataset>_pcp.yaml`. The latter config file is set in the `experiment/semantic/<dataset>.yaml` config files (and `<dataset>_nano.yaml` and `<dataset>_11g.yaml` experiment files)
- **⚠️ Breaking change** The on-device-transforms no longer build the `x` tensor, 
as it is now built by the `SPT` module (or `PartitionAndSemanticModule` when 
training the partition). This leads to the following changes in the 
`datamodule/semantic/<dataset>.yaml` configs:
  - Removed `NAGAddKeysTo(to='x')`.
  - The augmentations `NAGJitterKey`, `NAGDropoutColumns` and `NAGDropoutRows`, 
  previously applied to `x`, are now  applied directly to the NAG attributes.
  - Added hydra parameter `point_hf`, `post_cnn_point_hf` and `segment_hf` 
  @package model.net (cf. `model/semantic/spt.yaml`);
  - Added `point_hf`, `segment_hf` and `post_cnn_point_hf` attributes to `SPT`. 
  These control what is placed in `x` for the PointStage and for the stages 
  operating at the segment level. When using a CNN in the PointStage, 
  post_cnn_point_hf specifies which features to (re)concatenate with the CNN 
  features before they are processed by the PointStage MLP.
- **⚠️ Breaking change**: `RGB` is no longer added to `x` separately from other 
features. Therefore, the position of `'rgb'` in lists `point_hf` and 
`segment_hf` now matters. To ensure backward compatibility of your checkpoint, 
put `'rgb'` at the end of your lists to reproduce the previous behavior.
- Removing calls to `torch` legacy tensor constructors in favor of 
`torch.tensor()` and improving the handling of the working device upon 
tensor construction. Typically, to avoid construction on CPU followed by 
unnecessary CPU-GPU transfers
- Extended `scatter_pca` to optionally run with the `'eigh'` or `'eig'`
algorithms internally. However, both produce the same results but `'eigh'` is 
much faster so there is no good reason for using `'eig'`
- All dataset configs now make use of new GPU-based operations and 
minimize device transfers. This considerably accelerates preprocessing
- All dataset configs now make use of `GroundElevation(xy_grid=...)` to 
accelerate ground surface modeling
- `show()` now supports `max_points < 1` (i.e. no subsampling) and has a 
slightly changed logic ruled by `display` for visualizations without displaying
them in the GUI
- Better handling of devices in `CutPursuitPartition`. If the input `Data` is on
GPU, only the necessary tensors are moved to CPU to be passed to cut-pursuit, 
and any tensor creation is happening on the adequate device
- Removed the multiprocessing of S3DIS rooms upon raw data reading. This was 
causing issues for several users and using cuda memory for no reason. We 
consider over-optimizing the time to read raw dataset formats beyond the 
responsibility of this project
- Exposing more explicitly in the configs that 
`SampleRadiusSubgraphs(k_max=...)` implies a maximum number of nodes sampled in
each subgraph
- Factorized all transforms for SPT with `CutPurtsuitPartition` in 
`datamodule/semantic/default.yaml`. This does not change the behavior of any
transform.
- **⚠️ Breaking Change**: modified `install.sh` to use precompiled wheels for 
`pgeof`, `pycut-pursuit`, and `pygrid-graph` rather than compiling them at 
installation time. Kudos to [**Romain Janvier**](https://github.com/rjanvier) 
for these. This means that the import paths for these libraries changed 
in the code, so **users with an anterior version will need to update their 
conda environment by running**:
    ```bash
    # Install dependencies from pre-compiled wheels
    pip install -U pgeof pycut-pursuit pygrid-graph
    
    # (Optional) Remove obsolete locally-compiled dependencies
    rm -rf src/dependencies/grid_graph
    rm -rf src/dependencies/parallel_cut_pursuit
    ```

### Deprecated

### Fixed
- `SampleRadiusSubgraphs` was not taking the `batch` attribute of the input 
`Data` objects when searching for neighbors. This could result in neighborhoods
"bleeding" across batch items
- `SampleRadiusSubgraphs(r=-1, k=k, disjoint=True)` was returning a `NAGBatch` 
`k` of copies of the input. This was not the desirable behavior: we want the 
input to be returned untouched when `r<=0`
- `NAG.clone()`, `Data.clone()`, and `CSRData.clone()` were not always producing
"deep" copies but sometimes only "shallow" copies. Which defeats the purpose of
the `clone()` function. This has been fixed, so there should no longer be shared
memory between an object and its clone 
- PyG's `collate()` (and torch's `default_collate()`) mechanism will concatenate 
tensors based on the dtype of the first element of the list, when 
`num_workers > 0`. For this specific reason, we need now uniformize all tensors
to the adequate dtype in `BAtch.from_data_list()` before PyG's `collate()` is
called
- `listify_with_reference` checks that it is a `ListConfig` (list type produced 
by the hydra) in addition to being a `List`.
- Fixed state_dict of `WeightedFocalLoss` (bug in `MultiLoss.state_dict` that 
didn't retrieved `WeightedFocalLoss.state_dict`)
- add parentheses in `ConfusionMatrix.update` to fix operator precedence before `|`
- Setting `rich<=14.0` dependency constraint to circumvent lightning logging 
issues 
- Adding missing `laspy` installation instruction in the tutorial notebook
- `CutPursuitPartition` aggregates graph edge features by summing them with 
`to_trimmed(reduce='add')` instead of averaging them `to_trimmed(reduce='mean')`
- `scatter_pca` was producing correct eigenvectors but wrong eigenvalues, up to
a scaling factor. Fixed the covariance matrix scaling error.
- `scatter_pca` was running the eigendecomposition on CPU by default. Removed 
this, largely accelerating all GPU-based transforms using `scatter_pca`: 
`PointFeatures`, `SegmentFeatures`, `RadiusHorizontalGraph`
- `SampleRadiusSubgraphs` defaults to `knn_brute_force` instead of relying on 
FRNN. This addresses the strong latency of radius subgraph search, due to FRNN
not handling well the few-queries-but-high-k regime
- `save_confusion_matrix` now correctly normalizes the 
confusion matrix: precision is computed by normalizing by the sum of predictions 
for each predicted class, and recall by the sum of true labels for 
each target class.

### Removed
- `instantiate_transforms` from `src.transforms.__init__.py` no longer checks 
whether the _OUT_TYPE/_IN_TYPE of consecutive transforms match. This allows 
`Transform` objects with `_IN_TYPE = NAG` to accept `Data` inputs.

## \[2.1.0\] - 2024-11-07

### Added

- Added a [CITATION.cff](CITATION.cff)
- Added a [CHANGELOG.md](CHANGELOG.md)
- Added support for serialization of `CSRBatch`, `Batch` and `NAGBatch` objects
- Added support for inferring how to un-batch some `Batch` attributes, even if 
not present when `Batch.from_data_list()` was initially called
- Added more tools to `GroundElevation` for modeling non-planar ground surfaces 
- Added helper for S3DIS 6-fold metrics computation for semantic segmentation
- Moved to `pgeof==0.3.0`
- Released a Superpoint Transformer 🧑‍🏫 tutorial with 
[slides](media/superpoint_transformer_tutorial.pdf), 
[notebook](notebooks/superpoint_transformer_tutorial.ipynb),
and [video](https://www.youtube.com/watch?v=2qKhpQs9gJw)
- Added more documentation throughout the [docs](docs) and in the code
- Added some documentation for our [interactive visualization tool](docs/visualization.md)
- Added a new mechanism for tracking validation and test predictions on a 
specific batch of interest (or on the entire set) through training. The behavior
is controlled from the model config with the `track_val_every_n_epoch`, 
`track_val_idx`, and `track_test_idx`

### Changed

- **⚠️ Breaking Change**: modified the serialization behavior of the data 
structures. You will need to re-run all your already-existing datasets' 
preprocessing to adjust to the new serialization format. It would also be 
possible to write a script that would adjust the existing HDF5 files, but we do 
not provide such script
- Remove `SampleSubNodes` from the validation and test transforms to ensure the 
validation and test forward passes are deterministic

### Deprecated

### Fixed

- Fixed several bugs, some of which introduced by recent commits...
- Fixed some installation issues

### Removed
