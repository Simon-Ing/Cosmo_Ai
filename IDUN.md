
+ [Yum in User Space](https://stackoverflow.com/questions/36651091/how-to-install-packages-in-linux-centos-without-root-user-with-automatic-depen)

```
module purge
module load intel/2020b
module load SciPy
```

```
cmake -D CMAKE_C_COMPILER=icc -D CMAKE_CXX_COMPILER=icc ..
```
