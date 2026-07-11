#! /bin/sh
# (C) 2025: Hans Georg Schaathun <georg@schaathun.net> 

# Build script
# As of 28 July 2025, this script works on Ubuntu 22.04 and Debian.
# It does not work on Ubuntu 24.04



sh Setup/buildvenv.sh
. venv/build/bin/activate

rm -rf build

conan install . --output-folder=build --build=missing 


( cd build && \
  cmake .. -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release && \
  cmake --build . && \
  cmake --install . --prefix=../src
) 

# python CosmoSimPy/datagen.py
