#!/bin/sh
# (C) 2022,2025: Hans Georg Schaathun <georg@schaathun.net> 

# Build scripts for NTNU's IDUN cluster.
# This does not use conan, and relies on IDUN's module system for
# dependencies.

module purge
# module load OpenCV/4.5.3-foss-2021a-contrib
# module load CMake/3.20.1-GCCcore-10.3.0
# module load SymEngine/0.7.0-GCC-10.3.0
# module load SciPy-bundle/2021.05-foss-2021a

module load OpenCV/4.8.0-foss-2022b-contrib
module load SymEngine/0.11.2-gfbf-2022b
module load CMake/3.24.3-GCCcore-12.2.0
# # module load Python/3.9.5-GCCcore-10.3.0
module load SciPy-bundle/2023.02-gfbf-2022b

module list

sh Setup/buildvenv.sh
. venv/build/bin/activate

rm -rf build
mkdir build
cmake . -B build
cmake --build build

mkdir -p bin lib 
cmake --install build --prefix src

python CosmoSimPy/datagen.py
