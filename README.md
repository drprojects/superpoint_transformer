<div align="center">

# Superpoint Transformer

[![python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.2+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[//]: # ([![Paper]&#40;https://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)
[//]: # ([![Conference]&#40;https://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)


Official implementation for
<br>
<br>
[_Efficient 3D Semantic Segmentation with Superpoint Transformer_](https://arxiv.org/abs/2306.08045) (ICCV 2023)
<br>
[![arXiv](https://img.shields.io/badge/arxiv-2306.08045-b31b1b.svg)](https://arxiv.org/abs/2306.08045)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8042712.svg)](https://doi.org/10.5281/zenodo.8042712)
[![Project page](https://img.shields.io/badge/Project_page-8A2BE2)](https://drprojects.github.io/superpoint-transformer)
<br>
<br>
[_Scalable 3D Panoptic Segmentation As Superpoint Graph Clustering_](https://arxiv.org/abs/2401.06704) (3DV 2024 Oral)
<br>
[![arXiv](https://img.shields.io/badge/arxiv-2401.06704-b31b1b.svg)](https://arxiv.org/abs/2401.06704)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10689037.svg)](https://doi.org/10.5281/zenodo.10689037)
[![Project page](https://img.shields.io/badge/Project_page-8A2BE2)](https://drprojects.github.io/supercluster)
<br>
<br>
If you ❤️ or use this project, don't forget to give it a ⭐, it means a lot to us !
<br>
</div>

```
@article{robert2023spt,
  title={Efficient 3D Semantic Segmentation with Superpoint Transformer},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
```
@article{robert2024scalable,
  title={Scalable 3D Panoptic Segmentation as Superpoint Graph Clustering},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={Proceedings of the IEEE International Conference on 3D Vision},
  year={2024}
}
```

<br>

## 📌  Description

### Superpoint Transformer

<p align="center">
  <img width="80%" src="./media/teaser_spt.png">
</p>

**Superpoint Transformer (SPT)** is a superpoint-based transformer 🤖 architecture that efficiently ⚡ 
performs **semantic segmentation** on large-scale 3D scenes. This method includes a 
fast algorithm that partitions 🧩 point clouds into a hierarchical superpoint 
structure, as well as a self-attention mechanism to exploit the relationships 
between superpoints at multiple scales. 

<div align="center">

| ✨ SPT in numbers ✨ |
| :---: |
| 📊 **SOTA on S3DIS 6-Fold** (76.0 mIoU) |
| 📊 **SOTA on KITTI-360 Val** (63.5 mIoU) |
| 📊 **Near SOTA on DALES** (79.6 mIoU) | 
| 🦋 **212k parameters** ([PointNeXt](https://github.com/guochengqian/PointNeXt) ÷ 200, [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer) ÷ 40) | 
| ⚡ S3DIS training in **3h on 1 GPU** ([PointNeXt](https://github.com/guochengqian/PointNeXt) ÷ 7, [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer) ÷ 70) | 
| ⚡ **Preprocessing x7 faster than [SPG](https://github.com/loicland/superpoint_graph)** |

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-3d-semantic-segmentation-with-1/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=efficient-3d-semantic-segmentation-with-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-3d-semantic-segmentation-with-1/3d-semantic-segmentation-on-dales)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-dales?p=efficient-3d-semantic-segmentation-with-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-3d-semantic-segmentation-with-1/3d-semantic-segmentation-on-kitti-360)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-kitti-360?p=efficient-3d-semantic-segmentation-with-1)

</div>

### SuperCluster

<p align="center">
  <img width="80%" src="./media/teaser_supercluster.png">
</p>

**SuperCluster** is a superpoint-based architecture for **panoptic segmentation** of (very) large 3D scenes 🐘 based on SPT. 
We formulate the panoptic segmentation task as a **scalable superpoint graph clustering** task. 
To this end, our model is trained to predict the input parameters of a graph optimization problem whose solution is a panoptic segmentation 💡.
This formulation allows supervising our model with per-node and per-edge objectives only, circumventing the need for computing an actual panoptic segmentation and associated matching issues at train time.
At inference time, our fast parallelized algorithm solves the small graph optimization problem, yielding object instances 👥.
Due to its lightweight backbone and scalable formulation, SuperCluster can process scenes of unprecedented scale at once, on a single GPU 🚀, with fewer than 1M parameters 🦋.

<div align="center">

| ✨ SuperCluster in numbers ✨ |
| :---: |
| 📊 **SOTA on S3DIS 6-Fold** (55.9 PQ) |
| 📊 **SOTA on S3DIS Area 5** (50.1 PQ) |
| 📊 **SOTA on ScanNet Val** (58.7 PQ) |
| 📊 **FIRST on KITTI-360 Val** (48.3 mIoU) |
| 📊 **FIRST on DALES** (61.2 mIoU) |
| 🦋 **212k parameters** ([PointGroup](https://github.com/dvlab-research/PointGroup) ÷ 37) | 
| ⚡ S3DIS training in **4h on 1 GPU** | 
| ⚡ **7.8km²** tile of **18M** points in **10.1s** on **1 GPU** |

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-3d-panoptic-segmentation-with/panoptic-segmentation-on-s3dis)](https://paperswithcode.com/sota/panoptic-segmentation-on-s3dis?p=scalable-3d-panoptic-segmentation-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-3d-panoptic-segmentation-with/panoptic-segmentation-on-s3dis-area5)](https://paperswithcode.com/sota/panoptic-segmentation-on-s3dis-area5?p=scalable-3d-panoptic-segmentation-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-3d-panoptic-segmentation-with/panoptic-segmentation-on-scannetv2)](https://paperswithcode.com/sota/panoptic-segmentation-on-scannetv2?p=scalable-3d-panoptic-segmentation-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-3d-panoptic-segmentation-with/panoptic-segmentation-on-kitti-360)](https://paperswithcode.com/sota/panoptic-segmentation-on-kitti-360?p=scalable-3d-panoptic-segmentation-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/scalable-3d-panoptic-segmentation-with/panoptic-segmentation-on-dales)](https://paperswithcode.com/sota/panoptic-segmentation-on-dales?p=scalable-3d-panoptic-segmentation-with)

</div>

<br>

## 📰  Updates

- **28.02.2024** Major code release for **panoptic segmentation**, implementing 
**[_Scalable 3D Panoptic Segmentation As Superpoint Graph Clustering_](https://arxiv.org/abs/2401.06704)**.
This new version also implements long-awaited features such as lightning's
`predict()` behavior, **voxel-resolution and full-resolution prediction**.
Some changes in the dependencies and repository structure are **not 
backward-compatible**. If you were already using anterior code versions, this means we recommend re-installing your conda environment and re-running the preprocessing or your datasets❗
- **15.10.2023** Our paper **[_Scalable 3D Panoptic Segmentation As Superpoint Graph Clustering_](https://arxiv.org/abs/2401.06704)** was accepted for an **oral** presentation at **[3DV 2024](https://3dvconf.github.io/2024/)** 🥳
- **06.10.2023** Come see our poster for **[_Efficient 3D Semantic Segmentation with Superpoint Transformer_](https://arxiv.org/abs/2306.08045)** at **[ICCV 2023](https://iccv2023.thecvf.com/)**
- **14.07.2023** Our paper **[_Efficient 3D Semantic Segmentation with Superpoint Transformer_](https://arxiv.org/abs/2306.08045)** was accepted at **[ICCV 2023](https://iccv2023.thecvf.com/)** 🥳
- **15.06.2023** Official release 🌱

<br>

## 💻  Environment requirements
This project was tested with:
- Linux OS
- **64G** RAM
- NVIDIA GTX 1080 Ti **11G**, NVIDIA V100 **32G**, NVIDIA A40 **48G**
- CUDA 11.8 and 12.1
- conda 23.3.1

<br>

## 🏗  Installation
Simply run [`install.sh`](install.sh) to install all dependencies in a new conda environment 
named `spt`. 
```bash
# Creates a conda env named 'spt' env and installs dependencies
./install.sh
```

> **Note**: See the [Datasets page](docs/datasets.md) for setting up your dataset
> path and file structure.

<br>

### 🔩  Project structure
```
└── superpoint_transformer
    │
    ├── configs                   # Hydra configs
    │   ├── callbacks                 # Callbacks configs
    │   ├── data                      # Data configs
    │   ├── debug                     # Debugging configs
    │   ├── experiment                # Experiment configs
    │   ├── extras                    # Extra utilities configs
    │   ├── hparams_search            # Hyperparameter search configs
    │   ├── hydra                     # Hydra configs
    │   ├── local                     # Local configs
    │   ├── logger                    # Logger configs
    │   ├── model                     # Model configs
    │   ├── paths                     # Project paths configs
    │   ├── trainer                   # Trainer configs
    │   │
    │   ├── eval.yaml                 # Main config for evaluation
    │   └── train.yaml                # Main config for training
    │
    ├── data                      # Project data (see docs/datasets.md)
    │
    ├── docs                      # Documentation
    │
    ├── logs                      # Logs generated by hydra and lightning loggers
    │
    ├── media                     # Media illustrating the project
    │
    ├── notebooks                 # Jupyter notebooks
    │
    ├── scripts                   # Shell scripts
    │
    ├── src                       # Source code
    │   ├── data                      # Data structure for hierarchical partitions
    │   ├── datamodules               # Lightning DataModules
    │   ├── datasets                  # Datasets
    │   ├── dependencies              # Compiled dependencies
    │   ├── loader                    # DataLoader
    │   ├── loss                      # Loss
    │   ├── metrics                   # Metrics
    │   ├── models                    # Model architecture
    │   ├── nn                        # Model building blocks
    │   ├── optim                     # Optimization 
    │   ├── transforms                # Functions for transforms, pre-transforms, etc
    │   ├── utils                     # Utilities
    │   ├── visualization             # Interactive visualization tool
    │   │
    │   ├── eval.py                   # Run evaluation
    │   └── train.py                  # Run training
    │
    ├── tests                     # Tests of any kind
    │
    ├── .env.example              # Example of file for storing private environment variables
    ├── .gitignore                # List of files ignored by git
    ├── .pre-commit-config.yaml   # Configuration of pre-commit hooks for code formatting
    ├── install.sh                # Installation script
    ├── LICENSE                   # Project license
    └── README.md

```

> **Note**: See the [Datasets page](docs/datasets.md) for further details on `data/``. 

> **Note**: See the [Logs page](docs/logging.md) for further details on `logs/``. 

<br>

## 🚀  Usage
### Datasets
See the [Datasets page](docs/datasets.md) to set up your datasets. 

### Evaluation
Use the following command structure for evaluating our models from a checkpoint 
file `checkpoint.ckpt`, where `<task>` should be `semantic` for using SPT and `panoptic` for using 
SuperCluster:

```bash
# Evaluate for <task> segmentation on <dataset>
python src/eval.py experiment=<task>/<dataset> ckpt_path=/path/to/your/checkpoint.ckpt
```

Some examples:

```bash
# Evaluate SPT on S3DIS Fold 5
python src/eval.py experiment=semantic/s3dis datamodule.fold=5 ckpt_path=/path/to/your/checkpoint.ckpt

# Evaluate SPT on KITTI-360 Val
python src/eval.py experiment=semantic/kitti360  ckpt_path=/path/to/your/checkpoint.ckpt 

# Evaluate SPT on DALES
python src/eval.py experiment=semantic/dales ckpt_path=/path/to/your/checkpoint.ckpt

# Evaluate SuperCluster on S3DIS Fold 5
python src/eval.py experiment=panoptic/s3dis datamodule.fold=5 ckpt_path=/path/to/your/checkpoint.ckpt

# Evaluate SuperCluster on S3DIS Fold 5 with {wall, floor, ceiling} as 'stuff'
python src/eval.py experiment=panoptic/s3dis_with_stuff datamodule.fold=5 ckpt_path=/path/to/your/checkpoint.ckpt

# Evaluate SuperCluster on ScanNet Val
python src/eval.py experiment=panoptic/scannet ckpt_path=/path/to/your/checkpoint.ckpt

# Evaluate SuperCluster on KITTI-360 Val
python src/eval.py experiment=panoptic/kitti360  ckpt_path=/path/to/your/checkpoint.ckpt 

# Evaluate SuperCluster on DALES
python src/eval.py experiment=panoptic/dales ckpt_path=/path/to/your/checkpoint.ckpt
```

> **Note**: 
> 
> The pretrained weights of the **SPT** and **SPT-nano** models for 
>**S3DIS 6-Fold**, **KITTI-360 Val**, and **DALES** are available at:
>
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8042712.svg)](https://doi.org/10.5281/zenodo.8042712)
> 
> The pretrained weights of the **SuperCluster** models for 
>**S3DIS 6-Fold**, **S3DIS 6-Fold with stuff**, **ScanNet Val**,, **KITTI-360 Val**, and **DALES** are available at:
>
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10689037.svg)](https://doi.org/10.5281/zenodo.10689037)

### Training
Use the following command structure for **train our models on a 32G-GPU**, 
where `<task>` should be `semantic` for using SPT and `panoptic` for using 
SuperCluster:

```bash
# Train for <task> segmentation on <dataset>
python src/train.py experiment=<task>/<dataset>
```

Some examples:

```bash
# Train SPT on S3DIS Fold 5
python src/train.py experiment=semantic/s3dis datamodule.fold=5

# Train SPT on KITTI-360 Val
python src/train.py experiment=semantic/kitti360 

# Train SPT on DALES
python src/train.py experiment=semantic/dales

# Train SuperCluster on S3DIS Fold 5
python src/train.py experiment=panoptic/s3dis datamodule.fold=5

# Train SuperCluster on S3DIS Fold 5 with {wall, floor, ceiling} as 'stuff'
python src/train.py experiment=panoptic/s3dis_with_stuff datamodule.fold=5

# Train SuperCluster on ScanNet Val
python src/train.py experiment=panoptic/scannet

# Train SuperCluster on KITTI-360 Val
python src/train.py experiment=panoptic/kitti360 

# Train SuperCluster on DALES
python src/train.py experiment=panoptic/dales
```

Use the following to **train on a 11G-GPU 💾** (training time and performance 
may vary):

```bash
# Train SPT on S3DIS Fold 5
python src/train.py experiment=semantic/s3dis_11g datamodule.fold=5

# Train SPT on KITTI-360 Val
python src/train.py experiment=semantic/kitti360_11g 

# Train SPT on DALES
python src/train.py experiment=semantic/dales_11g

# Train SuperCluster on S3DIS Fold 5
python src/train.py experiment=panoptic/s3dis_11g datamodule.fold=5

# Train SuperCluster on S3DIS Fold 5 with {wall, floor, ceiling} as 'stuff'
python src/train.py experiment=panoptic/s3dis_with_stuff_11g datamodule.fold=5

# Train SuperCluster on ScanNet Val
python src/train.py experiment=panoptic/scannet_11g

# Train SuperCluster on KITTI-360 Val
python src/train.py experiment=panoptic/kitti360_11g 

# Train SuperCluster on DALES
python src/train.py experiment=panoptic/dales_11g
```

> **Note**: Encountering CUDA Out-Of-Memory errors 💀💾 ? See our dedicated 
> [troubleshooting section](#cuda-out-of-memory-errors).

> **Note**: Other ready-to-use configs are provided in
>[`configs/experiment/`](configs/experiment). You can easily design your own 
>experiments by composing [configs](configs):
>```bash
># Train Nano-3 for 50 epochs on DALES
>python src/train.py datamodule=dales model=nano-3 trainer.max_epochs=50
>```
>See 
>[Lightning-Hydra](https://github.com/ashleve/lightning-hydra-template) for more
>information on how the config system works and all the awesome perks of the 
> Lightning+Hydra combo.

> **Note**: By default, your logs will automatically be uploaded to 
>[Weights and Biases](https://wandb.ai), from where you can track and compare 
>your experiments. Other loggers are available in 
>[`configs/logger/`](configs/logger). See 
>[Lightning-Hydra](https://github.com/ashleve/lightning-hydra-template) for more
>information on the logging options.

### PyTorch Lightning `predict()`
Both SPT and SuperCluster inherit from `LightningModule` and implement `predict_step()`, which permits using 
[PyTorch Lightning's `Trainer.predict()` mechanism](https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html).

```python
from src.models.semantic import SemanticSegmentationModule
from src.datamodules.s3dis import S3DISDataModule
from pytorch_lightning import Trainer

# Predict behavior for semantic segmentation on S3DIS
dataloader = S3DISDataModule(...)
model = SemanticSegmentationModule(...)
trainer = Trainer(...)
batch, output = trainer.predict(model=model, dataloaders=dataloader)
```

This, however, still requires you to instantiate a `Trainer`, a `DataLoader`, 
and a model with relevant parameters.

For a little more simplicity, all our datasets inherit from 
`LightningDataModule` and implement `predict_dataloader()` by pointing to their 
corresponding test set by default. This permits directly passing a datamodule to
[PyTorch Lightning's `Trainer.predict()`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#predict)
without explicitly instantiating a `DataLoader`.

```python
from src.models.semantic import SemanticSegmentationModule
from src.datamodules.s3dis import S3DISDataModule
from pytorch_lightning import Trainer

# Predict behavior for semantic segmentation on S3DIS
datamodule = S3DISDataModule(...)
model = SemanticSegmentationModule(...)
trainer = Trainer(...)
batch, output = trainer.predict(model=model, datamodule=datamodule)
```

For more details on how to instantiate these, as well as the output format
of our model, we strongly encourage you to play with our 
[demo notebook](notebooks/demo.ipynb) and have a look at the [`src/eval.py`](src/eval.py) script.

### Full-resolution predictions
By design, our models only need to produce predictions for the superpoints of 
the $P_1$ partition level during training. 
All our losses and metrics are formulated as superpoint-wise objectives. 
This conveniently saves compute and memory at training and evaluation time.

At inference time, however, we often need the **predictions on the voxels** of the
$P_0$ partition level or on the **full-resolution input point cloud**.
To this end, we provide helper functions to recover voxel-wise and full-resolution
predictions.

See our [demo notebook](notebooks/demo.ipynb) for more details on these.

### Parameterizing SuperCluster graph clustering
One specificity of SuperCluster is that the model is not trained to explicitly 
do panoptic segmentation, but to predict the input parameters of a superpoint 
graph clustering problem whose solution is a panoptic segmentation.

For this reason, the hyperparameters for this graph optimization problem are 
selected after training, with a grid search on the training or validation set.
We find that fairly similar hyperparameters yield the best performance on all 
our datasets (see our [paper](https://arxiv.org/abs/2401.06704)'s appendix). Yet, you may want to explore 
these hyperparameters for your own dataset. To this end, see our 
[demo notebook](notebooks/demo_panoptic_parametrization.ipynb) for 
parameterizing the panoptic segmentation.

### Notebooks & visualization
We provide [notebooks](notebooks) to help you get started with manipulating our 
core data structures, configs loading, dataset and model instantiation, 
inference on each dataset, and visualization.

In particular, we created an interactive visualization tool ✨ which can be used
to produce shareable HTMLs. Demos of how to use this tool are provided in 
the [notebooks](notebooks). Additionally, examples of such HTML files are 
provided in [media/visualizations.7z](media/visualizations.7z)

<br>

## 📚  Documentation
- [README](README.md) - General introduction to the project
- [Data](docs/data_structures.md) - Introduction to `NAG` and `Data`, the core data structures of this project
- [Datasets](docs/datasets.md) - Introduction to `Datasets` and the project's `data/` structure
- [Logging](docs/logging.md) - Introduction to logging and the project's `logs/` structure

> **Note**: We endeavoured to **comment our code** as much as possible to make 
> this project usable. Still, if you find some parts are unclear or some more 
> documentation would be needed, feel free to let us know by creating an issue ! 

<br>

## 👩‍🔧  Troubleshooting
Here are some common issues and tips for tackling them.

### SPT or SuperCluster on an 11G-GPU 
Our default configurations are designed for a 32G-GPU. Yet, SPT and SuperCluster can run 
on an **11G-GPU 💾**, with minor time and performance variations.

We provide configs in [`configs/experiment/semantic`](configs/experiment/semantic) for 
training SPT on an **11G-GPU 💾**:

```bash
# Train SPT on S3DIS Fold 5
python src/train.py experiment=semantic/s3dis_11g datamodule.fold=5

# Train SPT on KITTI-360 Val
python src/train.py experiment=semantic/kitti360_11g 

# Train SPT on DALES
python src/train.py experiment=semantic/dales_11g
```

Similarly, we provide configs in [`configs/experiment/panoptic`](configs/experiment/panoptic) for 
training SuperCluster on an **11G-GPU 💾**:

```bash
# Train SuperCluster on S3DIS Fold 5
python src/train.py experiment=panoptic/s3dis_11g datamodule.fold=5

# Train SuperCluster on S3DIS Fold 5 with {wall, floor, ceiling} as 'stuff'
python src/train.py experiment=panoptic/s3dis_with_stuff_11g datamodule.fold=5

# Train SuperCluster on ScanNet Val
python src/train.py experiment=panoptic/scannet_11g

# Train SuperCluster on KITTI-360 Val
python src/train.py experiment=panoptic/kitti360_11g 

# Train SuperCluster on DALES
python src/train.py experiment=panoptic/dales_11g
```


### CUDA Out-Of-Memory Errors
Having some CUDA OOM errors 💀💾 ? Here are some parameters you can play 
with to mitigate GPU memory use, based on when the error occurs.

<details>
<summary><b>Parameters affecting CUDA memory.</b></summary>

**Legend**: 🟡 Preprocessing | 🔴 Training | 🟣 Inference (including validation and testing during training)

| Parameter                                   | Description                                                                                                                                                                                                                        |  When  |
|:--------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------:|
| `datamodule.xy_tiling`                      | Splits dataset tiles into xy_tiling^2 smaller tiles, based on a regular XY grid. Ideal square-shaped tiles à la DALES. Note this will affect the number of training steps.                                                         |  🟡🟣  |
| `datamodule.pc_tiling`                      | Splits dataset tiles into 2^pc_tiling smaller tiles, based on a their principal component. Ideal for varying tile shapes à la S3DIS and KITTI-360. Note this will affect the number of training steps.                             |  🟡🟣  |
| `datamodule.max_num_nodes`                  | Limits the number of $P_1$ partition nodes/superpoints in the **training batches**.                                                                                                                                                |   🔴   |
| `datamodule.max_num_edges`                  | Limits the number of $P_1$ partition edges in the **training batches**.                                                                                                                                                            |   🔴   |
| `datamodule.voxel`                          | Increasing voxel size will reduce preprocessing, training and inference times but will reduce performance.                                                                                                                         | 🟡🔴🟣 |
| `datamodule.pcp_regularization`             | Regularization for partition levels. The larger, the fewer the superpoints.                                                                                                                                                        | 🟡🔴🟣 |
| `datamodule.pcp_spatial_weight`             | Importance of the 3D position in the partition. The smaller, the fewer the superpoints.                                                                                                                                            | 🟡🔴🟣 |
| `datamodule.pcp_cutoff`                     | Minimum superpoint size. The larger, the fewer the superpoints.                                                                                                                                                                    | 🟡🔴🟣 |
| `datamodule.graph_k_max`                    | Maximum number of adjacent nodes in the superpoint graphs. The smaller, the fewer the superedges.                                                                                                                                  | 🟡🔴🟣 |
| `datamodule.graph_gap`                      | Maximum distance between adjacent superpoints int the superpoint graphs. The smaller, the fewer the superedges.                                                                                                                    | 🟡🔴🟣 |
| `datamodule.graph_chunk`                    | Reduce to avoid OOM when `RadiusHorizontalGraph` preprocesses the superpoint graph.                                                                                                                                                |   🟡   |
| `datamodule.dataloader.batch_size`          | Controls the number of loaded tiles. Each **train batch** is composed of `batch_size`*`datamodule.sample_graph_k` spherical samplings. Inference is performed on **entire validation and test tiles**, without spherical sampling. |  🔴🟣  |
| `datamodule.sample_segment_ratio`           | Randomly drops a fraction of the superpoints at each partition level.                                                                                                                                                              |   🔴   |
| `datamodule.sample_graph_k`                 | Controls the number of spherical samples in the **train batches**.                                                                                                                                                                 |   🔴   |
| `datamodule.sample_graph_r`                 | Controls the radius of spherical samples in the **train batches**. Set to `sample_graph_r<=0` to use the entire tile without spherical sampling.                                                                                   |   🔴   |
| `datamodule.sample_point_min`               | Controls the minimum number of $P_0$ points sampled per superpoint in the **train batches**.                                                                                                                                       |   🔴   |
| `datamodule.sample_point_max`               | Controls the maximum number of $P_0$ points sampled per superpoint in the **train batches**.                                                                                                                                       |   🔴   |
| `callbacks.gradient_accumulator.scheduling` | Gradient accumulation. Can be used to train with smaller batches, with more training steps.                                                                                                                                        |   🔴   |

<br>
</details>

<br>

## 💳  Credits
- This project was built using [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template).
- The main data structures of this work rely on [PyToch Geometric](https://github.com/pyg-team/pytorch_geometric)
- Some point cloud operations were inspired from the [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d), although not merged with the official project at this point. 
- For the KITTI-360 dataset, some code from the official [KITTI-360](https://github.com/autonomousvision/kitti360Scripts) was used.
- Some superpoint-graph-related operations were inspired from [Superpoint Graph](https://github.com/loicland/superpoint_graph)
- The hierarchical superpoint partition and graph clustering are computed using [Parallel Cut-Pursuit](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit)

<br>

## Citing our work
If your work uses all or part of the present code, please include the following a citation:

```
@article{robert2023spt,
  title={Efficient 3D Semantic Segmentation with Superpoint Transformer},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@article{robert2024scalable,
  title={Scalable 3D Panoptic Segmentation as Superpoint Graph Clustering},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={Proceedings of the IEEE International Conference on 3D Vision},
  year={2024}
}
```

You can find our [SPT paper 📄](https://arxiv.org/abs/2306.08045) and [SuperCluster paper 📄](https://arxiv.org/abs/2401.06704) on arxiv.

Also, if you ❤️ or use this project, don't forget to give it a ⭐, it means a lot to us !
