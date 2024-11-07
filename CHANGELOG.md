# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## \[2.1.0\] - 2024-11-07

### Added

- Added a [CITATION.cff](CITATION.cff)
- Added a [CHANGELOG.md](CHANGELOG.md)
- Added support for serialization of `CSRBatch`, `Batch` and `NAGBatch` objects
- Added support for inferring how to un-batch some `Batch` attributes, even if 
not present when `Batch.from_data_list()` was initially called
- Added helper for S3DIS 6-fold metrics computation for semantic segmentation
- Moved to `pgeof==0.3.0`
- Released a Superpoint Transformer 🧑‍🏫 tutorial with 
[slides](media/superpoint_transformer_tutorial.pdf), 
[notebook](notebooks/superpoint_transformer_tutorial.ipynb),
and [video](https://www.youtube.com/watch?v=2qKhpQs9gJw)
- Added more documentation throughout the [docs](docs) and in the code
- Added some documentation for our [interactive visualization tool](docs/visualization.md)

### Changed

- Breaking Change: modified the serialization behavior of the data structures.
You will need to re-run all your datasets' preprocessing
- Remove `SampleSubNodes` from the validation and test transforms to ensure the 
validation and test forward passes are deterministic

### Deprecated

### Fixed

- Fixed several bugs, some of which introduced by recent commits...
- Fixed some installation issues

### Removed
