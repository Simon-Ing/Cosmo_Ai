---
title: Running CosmoSim on IDUN
---

IDUN is the shared HPC cluster at NTNU, and these notes are thus
only intended for NTNU users.

The base system has been set up to use conan to manage dependencies,
but some conan packages depend on system libraries which cannot be
installed at IDUN.  

You can build with
```
sh Setup/idunbuild.sh
```

Assuming that the working directory is at `%HOME/git/CosmoSim`
the following gives a working runtime environment:
```
. Setup/idun.sh
```

This puts the module in the `PYTHONPATH`, so that you can run, e.g.
```
python -m CosmoSim.datagen
```

This approach does not make the module pip installable.


# Notes

## Intel Compiler

It seems that the Intel compiler is not compatible with the most
recent libraries.  We use open source compilers in our setup.

## Libraries in User space

It may be possible to install libraries in user space, but we
never went through with this.

+ [Yum in User Space](https://stackoverflow.com/questions/36651091/how-to-install-packages-in-linux-centos-without-root-user-with-automatic-depen)

```
export PATH="$HOME/local/usr/sbin:$HOME/local/usr/bin:$HOME/local/bin:$PATH"
export MANPATH="$HOME/local/usr/share/man:$MANPATH"
export LD_LIBRARY_PATH="$HOME/local/usr/lib:$HOME/local/usr/lib64:$LD_LIBRARY_PATH"
export LD_RUN_PATH="$LD_RUN_PATH:$HOME/local/usr/lib:$HOME/local/usr/lib64"
export CPATH=$HOME/local/usr/include:$CPATH
```


