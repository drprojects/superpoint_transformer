# -------------------------------------------------------------------- #
#        Distutils setup script for compiling python extensions        #
# -------------------------------------------------------------------- #
""" 
Compilation command: `python scripts/setup_dependencies.py build_ext`
Hugo Raguet, adapted by Loic Landrieu (2020), and Damien Robert (2022)
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
if os.name == 'nt':  # windows
    extra_compile_args = ["/std:c++11", "/openmp", "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = ["/lgomp"]
elif os.name == 'posix':  # linux
    extra_compile_args = ["-std=c++11", "-fopenmp", "-DMIN_OPS_PER_THREAD=10000"]
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

# Move the appropriate working directory
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

# Move the appropriate working directory
os.chdir(osp.join(DEPENDENCIES_DIR, 'parallel_cut_pursuit/python'))
name = "cp_kmpp_d0_dist_cpy"

if not osp.exists("bin"):
    os.mkdir("bin")

# Remove previously compiled lib
purge("bin/", name)

# Local modification to use uint32 instead of uint16 for components
# indexing. Simply put, this allows partitions of up to 4294967295
# components instead of 65535
filepath = osp.join('cpython', name + '.cpp')
assert osp.isfile(filepath)

with open(filepath, 'r') as f:
    lines = f.readlines()

for i_line, line in enumerate(lines):
    if 'typedef int16_t comp_t' in line:
        lines[i_line] = '    // typedef int16_t comp_t;\n'
        lines[i_line + 1] = '    // #define NPY_COMP NPY_INT16\n'
    if 'typedef uint16_t comp_t' in line:
        lines[i_line] = '    // typedef uint16_t comp_t;\n'
        lines[i_line + 1] = '    // #define NPY_COMP NPY_UINT16\n'
    if 'typedef int32_t comp_t' in line:
        lines[i_line] = '    typedef int32_t comp_t;\n'
        lines[i_line + 1] = '    #define NPY_COMP NPY_INT32\n'
    if 'typedef uint32_t comp_t' in line:
        lines[i_line] = '    typedef uint32_t comp_t;\n'
        lines[i_line + 1] = '    #define NPY_COMP NPY_UINT32\n'

with open(filepath, 'w') as f:
    f.writelines(lines)

# Compilation
mod = Extension(
    name,
    # list source files
    ["cpython/cp_kmpp_d0_dist_cpy.cpp", "../src/cp_kmpp_d0_dist.cpp",
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
