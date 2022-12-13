module purge
module load OpenCV/4.5.3-foss-2021a-contrib
module load CMake/3.20.1-GCCcore-10.3.0
module load SymEngine/0.7.0-GCC-10.3.0
mkdir -p bin lib build
cd build || exit 1
cmake ..
cmake --build .
cmake --install . --prefix ..
