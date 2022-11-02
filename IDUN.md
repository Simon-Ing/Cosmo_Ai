
+ [Yum in User Space](https://stackoverflow.com/questions/36651091/how-to-install-packages-in-linux-centos-without-root-user-with-automatic-depen)

```
module purge
module load intel/2020b
module load SciPy-bundle/2020.11-intel-2020b
module load CMake/3.18.4-GCCcore-10.2.0
```

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/local/usr/lib:$HOME/local/usr/lib64"
export LD_RUN_PATH="$LD_RUN_PATH:$HOME/local/usr/lib:$HOME/local/usr/lib64"
export CPATH=$CPATH:$HOME/local/usr/include
```

```
cmake -D CMAKE_C_COMPILER=icc -D CMAKE_CXX_COMPILER=icc ..
```
