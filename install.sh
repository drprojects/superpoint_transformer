#!/bin/bash

# Local variables
PROJECT_NAME=spt
PYTHON=3.8


# Recover the project's directory from the position of the install.sh
# script and move there. Not doing so would install some dependencies in
# the wrong place
HERE=`dirname $0`
HERE=`realpath $HERE`
cd $HERE


# Installation of Superpoint Transformer in a conda environment
echo "_____________________________________________"
echo
echo "         ☁ Superpoint Transformer 🤖         "
echo "                  Installer                  "
echo
echo "_____________________________________________"
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
    read -p "Please provide you conda install directory: " CONDA_DIR
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
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torchmetrics[detection]
#pip install torch==1.12.0 torchvision
pip install torch_geometric==2.3 pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
#pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.3 -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
pip install plyfile
pip install h5py
pip install colorhash
pip install seaborn
pip3 install numba
pip install pytorch-lightning
#pip install pytorch-lightning==1.8
pip install pyrootutils
pip install hydra-core --upgrade
pip install hydra-colorlog
pip install hydra-submitit-launcher
pip install rich
pip install torch_tb_profiler
pip install wandb
pip install gdown

#*********************************

echo
echo
echo "⭐ Installing FRNN"
echo
git clone --recursive https://github.com/lxxue/FRNN.git src/dependencies/FRNN

# install a prefix_sum routine first
cd src/dependencies/FRNN/external/prefix_sum
python setup.py install

# install FRNN
cd ../../ # back to the {FRNN} directory
python setup.py install
cd ../../../

echo
echo
echo "⭐ Installing Point Geometric Features"
echo
conda install -c omnia eigen3 -y
export EIGEN_LIB_PATH="$CONDA_PREFIX/include"
python -m pip install git+https://github.com/drprojects/point_geometric_features

echo
echo
echo "⭐ Installing Parallel Cut-Pursuit"
echo
# Clone parallel-cut-pursuit and grid-graph repos
git clone -b improve_merge https://gitlab.com/1a7r0ch3/parallel-cut-pursuit.git src/dependencies/parallel_cut_pursuit
git clone https://gitlab.com/1a7r0ch3/grid-graph.git src/dependencies/grid_graph

# Compile the projects
python scripts/setup_dependencies.py build_ext

echo
echo
echo "🚀 Successfully installed SPT"
