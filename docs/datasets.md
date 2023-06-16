# Datasets

All datasets inherit from the `torch_geometric` `Dataset` class, allowing for 
automated download (when allowed), preprocessing and inference-time transforms. 
See the [official documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)
for more details. 

## Supported datasets
- [S3DIS](http://buildingparser.stanford.edu/dataset.html)
- [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php)
- [DALES](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php)

## Structure of the `data/` directory 
<details>
<summary><b>Data directory structure.</b></summary>

Datasets are stored under the following structure:

```
└── data
    ├── dales                                         # Structure for DALES
    │   ├── DALESObjects.tar.gz                         # (optional) Downloaded zipped dataset
    │   ├── raw                                         # Raw dataset files
    │   │   └── {{train, test}}                           # DALES' split/tile.ply structure
    │   │       └── {{tile_name}}.ply
    │   └── processed                                   # Preprocessed data
    |       └── {{train, val, test}}                      # Dataset splits
    |           └── {{preprocessing_hash}}                  # Preprocessing folder
    │               └── {{tile_name}}.h5                      # Preprocessed tile file
    │    
    ├── kitti360                                      # Structure for KITTI-360
    │   ├── raw                                         # Raw dataset files
    │   │   ├── data_3d_semantics_test.zip              # (optional) Downloaded zipped test dataset
    │   │   ├── data_3d_semantics.zip                   # (optional) Downloaded zipped train dataset
    │   │   └── data_3d_semantics                       # Contains all raw train and test sequences
    │   │       └── {{sequence_name}}                     # KITTI-360's sequence/static/window.ply structure
    │   │           └── static
    │   │               └── {{window_name}}.ply
    │   └── processed                                   # Preprocessed data
    │       └── {{train, val, test}}                      # Dataset splits
    │           └── {{preprocessing_hash}}                  # Preprocessing folder
    │               └── {{sequence_name}}
    │                   └── {{window_name}}.h5                # Preprocessed window file
    │    
    └── s3dis                                         # Structure for S3DIS
        ├── Stanford3dDataset_v1.2.zip                  # (optional) Downloaded zipped dataset
        ├── raw                                         # Raw dataset files
        │   └── Area_{{1, 2, 3, 4, 5, 6}}                 # S3DIS's area/room/room.txt structure
        │       └── {{room_name}}  
        │           └── {{room_name}}.txt
        └── processed                                   # Preprocessed data
            └── {{train, val, test}}                      # Dataset splits
                └── {{preprocessing_hash}}                  # Preprocessing folder
                    └── Area_{{1, 2, 3, 4, 5, 6}}.h5          # Preprocessed Area file

```
</details>

> **Note**: **Already have the dataset on your machine ?** Save memory 💾 by 
> simply symlinking or copying the files to `data/<dataset_name>/raw/`, following the 
> [above-described `data/` structure](#structure-of-the-data-directory).

## Setting up your own `data/` and `logs/` paths
The `data/` and `logs/` directories will store all your datasets and training 
logs. By default, these are placed in the repository directory. 

Since this may take some space, or your heavy data may be stored elsewhere, you 
may specify other paths for these directories by creating a 
`configs/local/defaults.yaml` file containing the following:

```yaml
# @package paths

# path to data directory
data_dir: /path/to/your/data/

# path to logging directory
log_dir: /path/to/your/logs/
```

## Automatic download and preprocessing
Following `torch_geometric`'s `Dataset` behaviour:
- missing files in `data/<dataset_name>/raw` structure ➡ automatic download
- missing files in `data/<dataset_name>/processed` structure ➡ automatic preprocessing

However, some datasets require you to **_manually download_**
from their official webpage. For those, you will need to manually setup the 
[above-described `data/` structure](#structure-of-the-data-directory). 

<div align="center">

| Dataset | Automatic download |
| :--- | :---: |
| S3DIS | ❌ |
| KITTI-360 | ❌ |
| DALES | ✅ |

</div>

## Pre-transforms, transforms, on-device transforms

Pre-transforms are the functions making up the preprocessing. 
These are called only once and their output is saved in 
`data/<dataset_name>/processed/`. These typically encompass neighbor search and 
partition construction.

The transforms are called by the `Dataloaders` at batch-creation time. These 
typically encompass sampling and data augmentations and are performed on CPU, 
before moving the batch to the GPU.

On-device transforms, are transforms to be performed on GPU. These are 
typically compute intensive operations that could not be done once and for all 
at preprocessing time, and are too slow to be performed on CPU.

## Preprocessing hash
Different from `torch_geometric`, you can have **multiple 
preprocessed versions** of each dataset, identified by their preprocessing hash.

This hash will change whenever the preprocessing configuration 
(_i.e._ pre-transforms) is modified in an impactful way (_e.g._ changing the 
partition regularization). 

Modifications of the transforms and on-device 
transforms will not affect your preprocessing hash.

## Mini datasets
Each dataset has a "mini" version which only processes a portion of the data, to
speedup experimentation. To use it, set your the 
[dataset config](configs/datamodule) of your choice:
```yaml
mini: True
```

Or, if you are using the CLI, use the following syntax:
```shell script
# Train SPT on mini-DALES
python src/train.py experiment=dales +datamodule.mini=True
```

## Creating your own dataset
To create your own dataset, you will need to do the following:
- create `YourDataset` class inheriting from `src.datasets.BaseDataset`
- create `YourDataModule` class inheriting from `src.datamodules.DataModule`
- create `configs/datamodule/your_dataset.yaml` config 
 
Instructions are provided in the docstrings of those classes and you can get
inspiration from our code for S3DIS, KITTI-360 and DALES to get started. 

We suggest that your config inherits from `configs/datamodule/default.yaml`. See
`configs/datamodule/s3dis.yaml`, `configs/datamodule/kitti360.yaml`, and 
`configs/datamodule/dales.yaml` for inspiration.
