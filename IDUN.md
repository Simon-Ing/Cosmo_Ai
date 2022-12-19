---
title: Running CosmoSim on IDUN
---

IDUN is the shared HPC cluster at NTNU, and these notes are thus
only intended for NTNU users.

The base system has been set up to use conan to manage dependencies,
but some conan packages depend on system libraries which cannot be
installed at IDUN.  Therefore conan should not be used, and 
`CMakeLists.txt` has been set up with a conditional on the host
name, with a specific setup for IDUN.

The `idunbuild.sh` scripts builds on IDUN and installs the binaries
in the bin directory under the working directory:

# Notes

## Intel Compiler

It seems that the Intel compiler is not compatible with the most
recent libraries.

```
module purge
module load intel/2020b
module load SciPy-bundle/2020.11-intel-2020b
module load CMake/3.18.4-GCCcore-10.2.0
```

```
cmake -D CMAKE_C_COMPILER=icc -D CMAKE_CXX_COMPILER=icc ..
```

## Libraries in User space

+ [Yum in User Space](https://stackoverflow.com/questions/36651091/how-to-install-packages-in-linux-centos-without-root-user-with-automatic-depen)

```
export PATH="$HOME/local/usr/sbin:$HOME/local/usr/bin:$HOME/local/bin:$PATH"
export MANPATH="$HOME/local/usr/share/man:$MANPATH"
export LD_LIBRARY_PATH="$HOME/local/usr/lib:$HOME/local/usr/lib64:$LD_LIBRARY_PATH"
export LD_RUN_PATH="$LD_RUN_PATH:$HOME/local/usr/lib:$HOME/local/usr/lib64"
export CPATH=$HOME/local/usr/include:$CPATH
```


