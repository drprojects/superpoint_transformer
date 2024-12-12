#!/bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

echo "⭐ Check that torch_scatter has been installed with CUDA support".
python $SCRIPTPATH/check_torch_scatter_CUDA.py

echo "⭐ check that FRNN and prefix_sum are installed"
cd $SCRIPTPATH
cd ../src/dependencies/FRNN/
python tests/frnn_ratio_small.py

echo "⭐ check that pgeof is installed"
cd $SCRIPTPATH
cd ../src/dependencies/pgeof/
python tests/test_pgeof.py

echo "⭐ check that SuperPoint Transformers import works.  "
python $SCRIPTPATH/check_import_spt.py
