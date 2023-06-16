<div align="center">

# Superpoint Transformer

[![python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg&#41;]&#40;https://www.nature.com/articles/nature14539&#41;)
[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/paper/2020&#41;)


Official implementation for
<br>
[_Efficient 3D Semantic Segmentation with Superpoint Transformer_](http://arxiv.org/abs/2306.08045)
<br>
🚀⚡🔥
<br>

[![arXiv](https://img.shields.io/badge/arxiv-2306.08045-b31b1b.svg)](http://arxiv.org/abs/2306.08045)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8042712.svg)](https://doi.org/10.5281/zenodo.8042712)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-3d-semantic-segmentation-with/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=efficient-3d-semantic-segmentation-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-3d-semantic-segmentation-with/3d-semantic-segmentation-on-kitti-360)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-kitti-360?p=efficient-3d-semantic-segmentation-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-3d-semantic-segmentation-with/3d-semantic-segmentation-on-dales)](https://paperswithcode.com/sota/3d-semantic-segmentation-on-dales?p=efficient-3d-semantic-segmentation-with)

</div>

<p align="center">
    <img width="90%" src="./media/teaser.jpg">
</p>

<br>

## 📌  Description

SPT is a superpoint-based transformer 🤖 architecture that efficiently ⚡ 
performs semantic segmentation on large-scale 3D scenes. This method includes a 
fast algorithm that partitions 🧩 point clouds into a hierarchical superpoint 
structure, as well as a self-attention mechanism to exploit the relationships 
between superpoints at multiple scales. 

<div align="center">

| ✨ SPT in numbers ✨ |
| :---: |
| 📊 **SOTA on S3DIS 6-Fold** (76.0 mIoU) |
| 📊 **SOTA on KITTI-360 Val** (63.5 mIoU) |
| 📊 **Near SOTA on DALES** (79.6 mIoU) | 
| 🦋 **212k parameters** ([PointNeXt-XL](https://github.com/guochengqian/PointNeXt) ÷ 200, [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer) ÷ 40) | 
| ⚡ S3DIS training in **3h on 1 GPU** ([PointNeXt](https://github.com/guochengqian/PointNeXt) ÷ 7, [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer) ÷ 70) | 
| ⚡ **Preprocessing x7 faster than [SPG](https://github.com/loicland/superpoint_graph)** |

</div>

<br>

## 📰  Updates

- **15.06.2023 Official release** 🌱

<br>

## 📋  Environment requirements
This project was tested with:
- Linux OS
- CUDA 11.8 ([`torch-geometric`](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) does not support CUDA 12.0 yet)
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
### Dataset file structure
See the [Datasets page](docs/datasets.md) to setup your datasets. 

### Training SPT
Use the following commands to train SPT:
```bash
# Train SPT on S3DIS Fold 5
# ⚠️ S3DIS does not support automatic download, follow prompted instructions
python src/train.py experiment=s3dis datamodule.fold=5

# Train SPT on KITTI-360 Val
# ⚠️ KITTI-360 does not support automatic download, follow prompted instructions
python src/train.py experiment=kitti360 

# Train SPT on DALES
python src/train.py experiment=dales
```

> **Note**: Other ready-to-use configs are provided in
>[`configs/experiment/`](configs/experiment). You can easily design your own 
>experiments by composing [configs](configs):
>```bash
># Train Nano-3 for 50 epochs on DALES
>python src/train.py datamodule=dales model=nano-3 trainer.max_epochs=50
>```
>See 
>[Lightning-Hydra](https://github.com/ashleve/lightning-hydra-template) for more
>information.

> **Note**: By default, your logs will automatically be uploaded to 
>[Weights and Biases](https://wandb.ai), from where you can track and compare 
>your experiments. Other loggers are available in 
>[`configs/logger/`](configs/logger). See 
>[Lightning-Hydra](https://github.com/ashleve/lightning-hydra-template) for more
>information.

### Evaluating SPT
Use the following commands to evaluate SPT from a checkpoint file 
`checkpoint.ckpt`:
```bash
# Evaluate SPT on S3DIS Fold 5
python src/eval.py experiment=s3dis datamodule.fold=5 ckpt_path=checkpoint.ckpt

# Evaluate SPT on KITTI-360 Val
# ⚠️ KITTI-360 does not support automatic download, follow prompted instructions
python src/eval.py experiment=kitti360  ckpt_path=checkpoint.ckpt 

# Evaluate SPT on DALES
python src/eval.py experiment=dales ckpt_path=checkpoint.ckpt
```

> **Note**: The pretrained weights of the **SPT** and **SPT-nano** models for 
>**S3DIS 6-Fold**, **KITTI-360 Val**, and **DALES** are available at:
>
>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8042712.svg)](https://doi.org/10.5281/zenodo.8042712) 


### Notebooks & visualization
We provide [notebooks](notebooks) to help you get started with manipulating our 
core data structures, configs loading, dataset and model instantiation, 
inference on each dataset, and visualization.

In particular, we created an interactive visualization tool ✨ which can be used
to produce shareable HTMLs. Examples of how to use this tool are provided in 
the [notebooks](notebooks).

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

## 💳  Credits
- This project was built using [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template).
- The main data structures of this work rely on [PyToch Geometric](https://github.com/pyg-team/pytorch_geometric)
- Some point cloud operations were inspired from the [Torch-Points3D framework](https://github.com/nicolas-chaulet/torch-points3d), although not merged with the official project at this point. 
- For the KITTI-360 dataset, some code from the official [KITTI-360](https://github.com/autonomousvision/kitti360Scripts) was used.
- Some superpoint-graph-related operations were inspired from [Superpoint Graph](https://github.com/loicland/superpoint_graph)
- The hierarchical superpoint partition is computed using [Parallel Cut-Pursuit](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit)

<br>

## Citing our work
If your work uses all or part of the present code, please include the following a citation:

```
@inproceedings{robert2023spt,
  title={Efficient 3D Semantic Segmentation with Superpoint Transformer},
  author={Robert, Damien and Raguet, Hugo and Landrieu, Loic},
  journal={arXiv preprint arXiv:2306.08045},
  year={2023}
}
```

You can find our [paper on arxiv 📄](http://arxiv.org/abs/2306.08045).