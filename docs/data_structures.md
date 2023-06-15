# Data structures

The main data structures of this project are `Data` and `NAG`.

## `Data`
A `Data` object stores a single-level graph. 
It inherits from `torch_geometric`'s `Data` and has a similar behavior (see the
[official documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) 
for more on this). 

Important specificities of our `Data` object are:
- `Data.super_index` stores the parent's index for each node in `Data`
- `Data.sub` holds a `Cluster` object indicating the children of each node in `Data`
- `Data.to_trimmed()` works like `torch_geometric`'s `Data.coalesce()` with the additional constraint that (i,j) and (j,i) edges are considered duplicates
- `Data.save()` and `Data.load()` allow optimized, memory-friendly I/O operations
- `Data.select()` indexes the nodes à la numpy

## `NAG`
`NAG` (Nested Acyclic Graph) stores the hierarchical partition in the form of a 
list of `Data` objects.

Important specificities of our `Data` object are:
- `NAG[i]` returns a `Data` object holding the partition level `ì`
- `NAG.get_super_index()` returns the index mapping nodes from any level `i` to `j` with `i<j`
- `NAG.get_sampling()` produces indices for sampling the superpoints with certain constraints
- `NAG.save()` and `NAG.load()` allow optimized, memory-friedly I/O operations
- `NAG.select()` indexes the nodes of a specified partition level à la numpy and updates the rest of the `NAG` structure accordingly

## 🚀  Getting familiar with the data structures
See the `notebooks/demo_nag.ipyng` notebook to play with and visualize a 
provided `NAG`, without needing a whole dataset. 
