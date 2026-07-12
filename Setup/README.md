# Build scripts

These scripts have been created to document different build
scenarioes which have been tried and tested.

+ `build.sh` - local build, creating the C++ libraries as well
  as the dynamic library file for python
+ `idunbuild.sh` - build for the NTNU HPC cluster (idun)
    + This is analogous to `build.sh`


## Auxiliary scripts

+ Build the virtual environments
    + `buildvenv.sh`  (build environmet)
    + `venv.sh` (runtime environment - no longer used)
+ `idun.sh`
    + load the virtual environment for Idun, including the module packages.
