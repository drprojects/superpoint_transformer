# Data structures

This project relies on specific data structures for manipulating hierarchical 
partitions. 
As a basic user, you will most likely be exposed to `Data` and `NAG`.
As you go deeper into the project, you may need to familiarize yourself with 
`CSRData`, `Cluster`, `InstanceData`, and all the associated batching 
structures. 

## `Data`
A `Data` object stores a single-level graph. 
It inherits from `torch_geometric`'s `Data` and has a similar behavior (see the
[official documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) 
for more on this). 

Similar to `torch_geometric`'s `Data`:
- `Data.pos` holds node positions
- `Data.x` holds node features
- `Data.y` holds node labels
- `Data.edge_index` holds edges between nodes
- `Data.edge_attr` holds edge features

Important additional specificities of our `Data` object are:
- `Data.super_index` holds, for each node, the index of the parent node (ie 
superpoint) in the partition level above. Said otherwise, `super_index` 
allows mapping from $P_i$ to $P_{i+1}$.
- `Data.sub` holds a `Cluster` object indicating, for each node, the children 
nodes in the partition level below. Said otherwise, `sub` allows mapping from 
$P_i$ to $P_{i-1}$.
- `Data.to_trimmed()` works like `torch_geometric`'s `Data.coalesce()` with the
additional constraint that (i,j) and (j,i) edges are considered duplicates
- `Data.save()` and `Data.load()` allow optimized, memory-friendly I/O 
operations
- `Data.select()` indexes the nodes à la numpy
- `Data.show()` for interactive visualization (see 
[visualization documentation](docs/visualization.md))

The `Batch` class allows for batching `Data` objects together, while preserving
their advanced mechanisms without index collisions. 

## `NAG`
`NAG` (Nested Acyclic Graph) stores the hierarchical partition in the form of a 
list of `Data` objects.

Important specificities of our `Data` object are:
- `NAG[i]` returns a `Data` object holding the partition level `ì`
- `NAG.get_super_index()` returns the index mapping nodes from any level `i` to
`j` with `i<j`
- `NAG.get_sampling()` produces indices for sampling the superpoints with 
certain constraints
- `NAG.save()` and `NAG.load()` allow optimized, memory-friendly I/O operations
- `NAG.select()` indexes the nodes of a specified partition level à la numpy 
and updates the rest of the `NAG` structure accordingly
- `NAG.show()` for interactive visualization (see 
[visualization documentation](docs/visualization.md))

The `NAGBatch` class allows for batching `NAG` objects together, while 
preserving their advanced mechanisms without index collisions.

## `CSRData`
`CSRData` is the data structure we use for manipulating sparse information. 
This implements the [CSR (Compressed Sparse Row)](https://en.wikipedia.org/wiki/Sparse_matrix) representation. Simply 
put, this struture stores a list of tensor objects as `values`, and a `pointers`
1D tensor for indexing `values`. The `j`th tensor values related to element `i`,
are stored in: `CSRData.values[j][CSRData.pointers[i]:CSRData.pointers[i+1]`.

The `CSRData` structure is designed for conveniently storing, indexing, 
updating, and batching data of varying sizes such as lists of lists while making
use of efficient, vectorized, torch operations. 

`CSRBatch` is the datastructure for batching multiple `CSRData` objects 
together.

## `Cluster`
`Cluster` inherits from `CSRData` and is specifically designed for storing 
cluster indices to efficiently gather indices of elements in a cluster `i`, 
given `i`. This structure could typically be implemented as a list of lists, 
but our implementation allows much faster parallelized operations for indexing,
updating, and batching this data.

When two `Cluster` are batched in an `ClusterBatch`, the indices will be updated
to avoid collision between the batch items.

## `InstanceData`
`InstanceData` inherits from `CSRData` and is specifically designed for 
manipulating instance and panoptic segmentation data. By convention, the 
element `i` by which we index the `values` is a point/superpoint/predicted 
instance. The associated `values` describe overlaps between the `i`th element
and the target instance objects of the dataset. These `values` include the 
following:

- `obj`: the index of the target instance object with which element `i` overlaps
- `count`: the number of points in the overlap
- `y`: the target instance object's semantic label

Importantly, each target object in the `InstanceData` is expected to be 
described by a unique index in `obj`, regardless of its actual semantic class.
It is not required for the labels of object instances in `obj` to be contiguous
in `[0, obj_max]`, although enforcing it may have beneficial downstream effects
on memory and I/O times.

Finally, when two `InstanceData` are batched in an `InstanceBatch`, the `obj` 
indices will be updated to avoid collision between the batch items.

## 🚀  Getting familiar with the data structures
See the [`notebooks/demo_nag.ipynb`](../notebooks/demo_nag.ipynb) notebook to play with and visualize a 
provided `NAG`, without needing a whole dataset.

For more information, have a look at the docstrings and code of each data 
structure, these are fairly commented and should help you gain a deeper 
understanding of their mechanisms.
