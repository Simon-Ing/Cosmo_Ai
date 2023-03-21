# (C) 2022: Hans Georg Schaathun <georg@schaathun.net> 

# Build script

rm -rf build
conan install . -if build
cmake . -B build
cmake --build build || exit 2

mkdir -p bin lib 
cmake --install build --prefix .
