#!/bin/sh
# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

# Build scripts for NTNU's IDUN cluster.

module purge
module load OpenCV/4.5.3-foss-2021a-contrib
module load CMake/3.20.1-GCCcore-10.3.0
module load SymEngine/0.7.0-GCC-10.3.0
# module load Python/3.9.5-GCCcore-10.3.0
module load SciPy-bundle/2021.05-foss-2021a
module list

rm -rf build
mkdir build
cmake . -B build
cmake --build build

mkdir -p bin lib 
cmake --install build --prefix .
