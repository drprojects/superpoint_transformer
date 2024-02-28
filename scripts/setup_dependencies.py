# -------------------------------------------------------------------- #
#        Distutils setup script for compiling python extensions        #
# -------------------------------------------------------------------- #
""" 
Compilation command: `python scripts/setup_dependencies.py build_ext`
Camille Baudoin and Hugo Raguet (2019), adapted by Loic Landrieu (2020), and Damien Robert (2022)
Source: https://github.com/loicland/img2svg
"""

from distutils.core import setup, Extension
from distutils.command.build import build
import numpy
import shutil
import os
import os.path as osp
import re


########################################################################
#                     Targets and compile options                      #
########################################################################

# Keep track of directories of interest
WORK_DIR = osp.realpath(os.curdir)
PROJECT_DIR = osp.realpath(osp.dirname(osp.dirname(__file__)))
DEPENDENCIES_DIR = osp.join(PROJECT_DIR, 'src', 'dependencies')

# Find the Numpy headers
include_dirs = [numpy.get_include(), "../include"]

# Compilation and linkage options
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
# COMP_T_ON_32_BITS for components identifiers on 32 bits rather than 16
if os.name == 'nt':  # windows
    extra_compile_args = ["/std:c++11", "/openmp",
        "-DMIN_OPS_PER_THREAD=10000", "-DCOMP_T_ON_32_BITS"]
    extra_link_args = ["/lgomp"]
elif os.name == 'posix':  # linux
    extra_compile_args = ["-std=c++11", "-fopenmp",
        "-DMIN_OPS_PER_THREAD=10000", "-DCOMP_T_ON_32_BITS"]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not supported yet.')


########################################################################
#                         Auxiliary functions                          #
########################################################################

class build_class(build):
    def initialize_options(self):
        build.initialize_options(self)
        self.build_lib = "bin"

    def run(self):
        build_path = self.build_lib


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(osp.join(dir, f))


########################################################################
#                              Grid graph                              #
########################################################################

# Move to the appropriate working directory
os.chdir(osp.join(DEPENDENCIES_DIR, 'grid_graph/python'))
name = "grid_graph"
if not osp.exists("bin"):
    os.mkdir("bin")

# Remove previously compiled lib
purge("bin/", name)

# Compilation
mod = Extension(
    name,
    # list source files
    ["cpython/grid_graph_cpy.cpp",
     "../src/edge_list_to_forward_star.cpp",
     "../src/grid_to_graph.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args)

setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))

# Postprocessing
try:
    # remove temporary compilation products
    shutil.rmtree("build")
except FileNotFoundError:
    pass

########################################################################
#                         Parallel cut-pursuit                         #
########################################################################

# Move to the appropriate working directory
os.chdir(osp.join(DEPENDENCIES_DIR, 'parallel_cut_pursuit/python'))
name = "cp_d0_dist_cpy"

if not osp.exists("bin"):
    os.mkdir("bin")

# Remove previously compiled lib
purge("bin/", name)

# Compilation
mod = Extension(
    name,
    # list source files
    ["cpython/cp_d0_dist_cpy.cpp", "../src/cp_d0_dist.cpp",
     "../src/cut_pursuit_d0.cpp", "../src/cut_pursuit.cpp",
     "../src/maxflow.cpp"],
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args)

setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))

# Postprocessing
try:
    # remove temporary compilation products
    shutil.rmtree("build")
except FileNotFoundError:
    pass

# Restore the initial working directory
os.chdir(WORK_DIR)