# Datasets

All datasets inherit from the `torch_geometric` `Dataset` class, allowing for 
automated preprocessing and inference-time transforms. 
See the [official documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)
for more details. 

## Supported datasets
<div align="center">

| Dataset                                                                                                                 |                                                           Download from ?                                                            | Which files ?                                        | Where to ? |
|:------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------|:----|
| [S3DIS](http://buildingparser.stanford.edu/dataset.html)                                                                |         [link](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1)          | `Stanford3dDataset_v1.2.zip`                         | `data/s3dis/` |
| [ScanNetV2](http://www.scan-net.org/)                                                                |         [link](http://www.scan-net.org/ScanNet)          | `scannetv2-labels.combined.tsv` `{{scan_name}}.aggregation.json` `{{scan_name}}.txt` `{{scan_name}}_vh_clean_2.0.010000.segs.json` `{{scan_name}}_vh_clean_2.ply` | `data/scannet/` |
| [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/index.php)                                                        |                                    [link](http://www.cvlibs.net/datasets/kitti-360/download.php)                                     | `data_3d_semantics.zip` `data_3d_semantics_test.zip` | `data/kitti360/` |
| [DALES](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) | [link](https://docs.google.com/forms/d/e/1FAIpQLSefhHMMvN0Uwjnj_vWQgYSvtFOtaoGFWsTIcRuBTnP09NHR7A/viewform?fbzx=5530674395784263977) | `DALESObjects.tar.gz`                                | `data/dales/` |

</div>


### Structure of the `data/` directory
<details>
<summary><b>S3DIS data directory structure.</b></summary>
<br><br>

```
└── data
    └── s3dis                                                     # Structure for S3DIS
        ├── Stanford3dDataset_v1.2.zip                              # (optional) Downloaded zipped dataset with non-aligned rooms
        ├── raw                                                     # Raw dataset files
        │   └── Area_{{1, 2, 3, 4, 5, 6}}                             # S3DIS's area/room/room.txt structure
        │       └── Area_{{1, 2, 3, 4, 5, 6}}_alignmentAngle.txt        # Room alignment angles required for entire floor reconstruction
        │       └── {{room_name}}  
        │           └── {{room_name}}.txt
        └── processed                                               # Preprocessed data
            └── {{train, val, test}}                                  # Dataset splits
                └── {{preprocessing_hash}}                              # Preprocessing folder
                    └── Area_{{1, 2, 3, 4, 5, 6}}.h5                      # Preprocessed Area file

```

> **Warning**: Make sure you download `Stanford3dDataset_v1.2.zip` and 
> **NOT** the aligned version ⛔ `Stanford3dDataset_v1.2_Aligned_Version.zip`,
> which does not contain the `Area_{{1, 2, 3, 4, 5, 6}}_alignmentAngle.txt` 
> files.

<br>
</details>

<details>
<summary><b>ScanNetV2 data directory structure.</b></summary>
<br><br>

```
└── data
    └─── scannet                                     # Structure for ScanNetV2
        ├── raw                                         # Raw dataset files
        |   ├── scannetv2-labels.combined.tsv
        |   ├── scans
        |   │   └── {{scan_name}}
        |   │       ├── {{scan_name}}.aggregation.json
        |   │       ├── {{scan_name}}.txt
        |   │       ├── {{scan_name}}_vh_clean_2.0.010000.segs.json
        |   │       └── {{scan_name}}_vh_clean_2.ply
        |   └── scans_test
        |       └── {{scan_name}}
        |           └── {{scan_name}}_vh_clean_2.ply
        └── processed                                   # Preprocessed data
            └── {{train, val, test}}                      # Dataset splits
                └── {{preprocessing_hash}}                  # Preprocessing folder
                    └── {{scans, scans_test}}
                        └── {{scan_name}}.h5                # Preprocessed scan file

```
<br>
</details>

<details>
<summary><b>KITTI-360 data directory structure.</b></summary>
<br><br>

```
└── data
    └─── kitti360                                     # Structure for KITTI-360
        ├── data_3d_semantics_test.zip                  # (optional) Downloaded zipped test dataset
        ├── data_3d_semantics.zip                       # (optional) Downloaded zipped train dataset
        ├── raw                                         # Raw dataset files
        │   └── data_3d_semantics                       # Contains all raw train and test sequences
        │       └── {{sequence_name}}                     # KITTI-360's sequence/static/window.ply structure
        │           └── static
        │               └── {{window_name}}.ply
        └── processed                                   # Preprocessed data
            └── {{train, val, test}}                      # Dataset splits
                └── {{preprocessing_hash}}                  # Preprocessing folder
                    └── {{sequence_name}}
                        └── {{window_name}}.h5                # Preprocessed window file

```
<br>
</details>

<details>
<summary><b>DALES data directory structure.</b></summary>
<br><br>

```
└── data
    └── dales                                         # Structure for DALES
        ├── DALESObjects.tar.gz                         # (optional) Downloaded zipped dataset
        ├── raw                                         # Raw dataset files
        │   └── {{train, test}}                           # DALES' split/tile.ply structure
        │       └── {{tile_name}}.ply
        └── processed                                   # Preprocessed data
            └── {{train, val, test}}                      # Dataset splits
                └── {{preprocessing_hash}}                  # Preprocessing folder
                    └── {{tile_name}}.h5                      # Preprocessed tile file

```

> **Warning**: Make sure you download the `DALESObjects.tar.gz` and **NOT** 
> ⛔ `dales_semantic_segmentation_las.tar.gz` nor 
> ⛔ `dales_semantic_segmentation_ply.tar.gz` versions, which do not contain 
> all required point attributes.

<br>
</details>

> **Note**: **Already have the dataset on your machine ?** Save memory 💾 by 
> simply symlinking or copying the files to `data/<dataset_name>/raw/`, following the 
> [above-described `data/` structure](#structure-of-the-data-directory).

### Automatic download and preprocessing
Following `torch_geometric`'s `Dataset` behaviour:

0. Dataset instantiation <br>
➡ Load preprocessed data in `data/<dataset_name>/processed`
1. Missing files in `data/<dataset_name>/processed` structure<br>
➡ **Automatic** preprocessing using files in `data/<dataset_name>/raw`
2. Missing files in `data/<dataset_name>/raw` structure<br>
➡ **Automatic** unzipping of the downloaded dataset in `data/<dataset_name>`
3. Missing downloaded dataset in `data/<dataset_name>` structure<br>
➡ ~~**Automatic**~~ **manual** download to `data/<dataset_name>`

> **Warning**: We **do not support ❌ automatic download**, for compliance 
>reasons.
>Please _**manually download**_ the required dataset files to the required 
>location as indicated in the above [table](#supported-datasets).


## Setting up your own `data/` and `logs/` paths
The `data/` and `logs/` directories will store all your datasets and training 
logs. By default, these are placed in the repository directory. 

Since this may take some space, or your heavy data may be stored elsewhere, you 
may specify other paths for these directories by creating a 
`configs/local/default.yaml` file containing the following:

```yaml
# @package paths

# path to data directory
data_dir: /path/to/your/data/

# path to logging directory
log_dir: /path/to/your/logs/
```

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
speedup experimentation. To use it, set the
[dataset config](../configs/datamodule) of your choice:
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
- create `configs/datamodule/<TASK>/your_dataset.yaml` config 
 
Instructions are provided in the docstrings of those classes, and you can get
inspiration from our code for S3DIS, ScanNet, KITTI-360 and DALES to get started. 

We suggest that your config inherits from `configs/datamodule/<TASK>/default.yaml`, where `<TASK>` is be `semantic` or `panoptic`, depending on your segmentation task of interest. See
`configs/datamodule/<TASK>/s3dis.yaml`, `configs/datamodule/<TASK>/scannet.yaml`, `configs/datamodule/<TASK>/kitti360.yaml`, and 
`configs/datamodule/<TASK>/dales.yaml` for inspiration.

#### Semantic label format
The semantic labels of your dataset must follow certain rules. 

Indeed, your points are expected to have labels within $[0, C]$, where: $C$ is 
the `num_classes` you define in your `YourDataset`. 

- **All labels $[0, C - 1]$ are assumed to be present in your dataset**. As 
such, they will all be used in metrics and losses computation.
- A point with the **$C$ label will be considered void/ignored/unlabeled** 
(whichever you call it). As such, it will be excluded from from metrics and 
losses computation

Hence, make sure the **output of your `YourDataset.read_single_raw_cloud()` 
reader method never returns labels outside of your $[0, C]$ range**. Besides, 
if some labels in $[0, C - 1]$ are not useful to you (ie absent from your 
dataset), we recommend you remap your labels to a new $[0, C' - 1]$ range
(`torch_geometric.nn.pool.consecutive.consecutive_cluster` can help you with 
that, if need be), while making sure you only use the label $C'$ for
void/ignored/unlabeled points. 

#### Setting your `train`, `val`, and `test` sets
The clouds you use for your respective sets are to be specified in the 
`all_base_cloud_ids()` method of your `YourDataset`.

```python
def all_base_cloud_ids(self):
    return {
        'train': [...],  # list of UNIQUE clouds ids in your train set
        'val': [...],  # list of UNIQUE clouds ids in your validation set
        'test': [...]  # list of UNIQUE clouds ids in your test set
    }
```

Importantly, the cloud ids specified in each split must be uniquely identified: 
we do not want clouds to have the same name in your `train` and `test` set.

#### Using the `test` set for validation
Generally, if you intend to run multiple experiments and tune some 
hyperparameters to suit your dataset, you do need a `validation` set to avoid
contaminating your `test` set, which must be kept aside until final performance
evaluation. Yet, in some cases you might want to only use a `train` and a `test`
set. In this case you must set:

```python
def all_base_cloud_ids(self):
    return {
        'train': [...],  # list of UNIQUE clouds ids in your train set
        'val': [],  # empty list, no validation clouds
        'test': [...]  # list of UNIQUE clouds ids in your test set
    }
```

Still, you can specify that you want to also use the `test` set as a `val` set 
(which is dangerous ML practice) by setting in your 
`configs/datamodule/your_task/your_dataset.yaml` datamodule config:

```yaml
val_on_test: True
```

#### `val` points mixed with `train`/`test` points
It sometimes happens that your validation points are stored in the same 
preprocessed files as your training or testing points. In this peculiar situation, it is 
possible to load the relevant files when needed and slice only the required points 
as an `on_device_transform` to save time. 

In this case, the `all_base_cloud_ids()` method of your `YourDataset` may contain duplicate
entries between `val` and the other splits:

```python
def all_base_cloud_ids(self):
    return {
        'train': [...],  # list of clouds ids in your train set, may contain duplicates with val
        'val': [...],  # list of clouds ids in your vallidation set
        'test': [...]  # list of clouds ids in your test set, may contain duplicates with val
    }
```

You must specify one of the following in your 
`configs/datamodule/your_task/your_dataset.yaml` datamodule config:

```yaml
val_mixed_in_train: True  # if some preprocessed clouds contain both validation and train points
test_mixed_in_val: True  # if some preprocessed clouds contain both validation and test points
```

Finally, your `read_single_raw_cloud()` method must return `Data` objects holding a `is_val` 
boolean attribute indicating whether a point belongs to the validation set. If `val_mixed_in_train`
or `test_mixed_in_val` are specified, this attribute will be used for selecting the relevant 
points at batch creation time. See S3DIS's `read_s3dis_area()` for an example of how `is_val` 
can be specified.
