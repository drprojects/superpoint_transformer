#!/bin/bash

# Local variables
PROJECT_NAME=spt
PYTHON=3.8
TORCH=2.2.0
CUDA_SUPPORTED=(11.8 12.1)

# Check if torchsparse should be installed
INSTALL_TORCHSPARSE=false
if [[ "$1" == "with_torchsparse" ]]; then
    INSTALL_TORCHSPARSE=true
elif [[ -n "$1" ]]; then
    echo "Unknown argument: $1"
    echo "Usage: ./install.sh [with_torchsparse]"
    echo "  with_torchsparse: Install TorchSparse (optional dependency)"
    exit 1
fi


# Recover the project's directory from the position of the install.sh
# script and move there. Not doing so would install some dependencies in
# the wrong place
HERE=`dirname $0`
HERE=`realpath $HERE`
cd $HERE


# Installation of Superpoint Transformer in a conda environment
echo "_____________________________________________"
echo
echo "         🧩 Superpoint Transformer 🤖        "
echo "                  Installer                  "
echo
echo "_____________________________________________"
echo
echo
echo "⭐ Searching for installed CUDA"
echo
# Recover the CUDA version using nvcc
CUDA_VERSION=`nvcc --version | grep release | sed 's/.* release //' | sed 's/, .*//'`
CUDA_MAJOR=`echo ${CUDA_VERSION} | sed 's/\..*//'`
CUDA_MINOR=`echo ${CUDA_VERSION} | sed 's/.*\.//'`

# If CUDA version not supported, print error and exit
if [[ ! " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_VERSION} " ]]
then
    echo "Found CUDA ${CUDA_VERSION} installed, which is not among the supported versions: "`echo ${CUDA_SUPPORTED[*]}`
    echo "Please update CUDA to one of the supported versions."
    exit 1
fi

echo
echo
echo "⭐ Searching for installed conda"
echo
# Recover the path to conda on your machine
# First search the default '~/miniconda3' and '~/anaconda3' paths. If
# those do not exist, ask for user input
CONDA_DIR=`realpath ~/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  CONDA_DIR=`realpath ~/anaconda3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide your conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda conda found at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh

echo
echo
echo "⭐ Creating conda environment '${PROJECT_NAME}'"
echo
# Create deep_view_aggregation environment from yml
conda create --name ${PROJECT_NAME} python=${PYTHON} -y

# Activate the env
source ${CONDA_DIR}/etc/profile.d/conda.sh  
conda activate ${PROJECT_NAME}

echo
echo
echo "⭐ Installing conda and pip dependencies"
echo
conda install pip nb_conda_kernels -y
pip install matplotlib
pip install plotly==5.9.0
pip install "jupyterlab>=3" "ipywidgets>=7.6" jupyter-dash
pip install "notebook>=5.3" "ipywidgets>=7.5"
pip install ipykernel
pip3 install torch==${TORCH} torchvision --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}
pip install torchmetrics==0.11.4
pip install pyg_lib torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-${TORCH}+cu${CUDA_MAJOR}${CUDA_MINOR}.html
pip install torch_geometric==2.3.0
pip install plyfile
pip install h5py
pip install colorhash
pip install seaborn
pip install numba
pip install pytorch-lightning
pip install pyrootutils
pip install hydra-core --upgrade
pip install hydra-colorlog
pip install hydra-submitit-launcher
pip install "rich<=14.0"
pip install torch_tb_profiler
pip install wandb
pip install open3d
pip install gdown
pip install ipyfilechooser
pip install torch-ransac3d
pip install pgeof
pip install pycut-pursuit
pip install pygrid-graph
pip install torch-graph-components

echo
echo
echo "⭐ Installing FRNN"
echo
git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN

# install a prefix_sum routine first
cd src/dependencies/FRNN/external/prefix_sum
pip install .

# install FRNN
cd ../../ # back to the {FRNN} directory
pip install .
cd ../../../

# install TorchSparse (optional)
if [[ "$INSTALL_TORCHSPARSE" == true ]]; then
    echo
    echo
    echo "⭐ Installing TorchSparse"
    echo
    git clone https://github.com/mit-han-lab/torchsparse.git src/dependencies/torchsparse
    pip install backports.cached-property
    pip install rootpath
    conda install -y google-sparsehash -c bioconda
    cd src/dependencies/torchsparse
    pip install .
    cd ../../../
fi

# let user know
echo
echo
echo "🚀 Successfully installed SPT"
