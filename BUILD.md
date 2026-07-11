# Building CosmoSim

A number of approaches have been attempted for building.
Some work, and others may serve as inspiration if you have
other needs. It is easiest to look at the various scripts 
and workflows that are included.

The currently working approaches are 
+ github workflow `wheels.yml`, building python wheels (pip installable)
  using `cibuildwheel` (Linux, macOS, and Windows)
+ `python -m build` building the python library locally using `skbuild-conan`
+ `Setup/build.sh` building locally, including both python and C++ libraries,
  working on MacOS and Linux (Debian&Ubuntu)
    + github workflow `regressiontest.yml` demonstrate the same build.
    + It depends on many preinstalled packages, but it (conan) will usually
      tell you which
+ If OpenCV and symengine (C++ libraries) are preinstalled, it should
  be possible to build with cmake, not using conan.
    + `Setup/idunbuild.sh` does this for the NTNU HPC cluster.
      It can be used as an example, but as written, it depends on their
      module system for preinstalled libraries.
Windows wheels are built with `cibuildwheel` via the `wheels.yml` workflow
and the `[tool.cibuildwheel.windows]` section of `pyproject.toml`, which
disables OpenCV's ffmpeg/video support so the build avoids the flaky
from-source ffmpeg/libaom compile on MSVC.

For broken or otherwise non-supported approaches, see
+ directory `docker`
+ directory `Setup`

## MacOS Desktop Application

The GUI is also available as desktop applications for MacOS 12 on 
Intel and MacOS 14 on arm64.  The desktop application integrates its
own Python interpreter, so that you do not have to install Python
separately.  However, you will have to manually assure the Mac that
you can trust the software.

The binaries are not signed, and you will have to confirm
that you trust the binary before it will run.

## Building from Source

The build procedure is primarily developed on Debian Bullseye, but it now 
also works reliably on github runners running Windows, Ubuntu, and MacOS.
We have limited capacity to develop generic and robust build procedures,
but we shall be happy to incorporate contributions.

The following instructions are Linux specific.

### Using conan

To install using conan, you can use included `build.sh` script,
which requires the setup of a virtual environment.
```sh
python -m venv pythonenv
. pythonenv/bin/activate
pip3 install -r requirements.txt
conan profile detect
sh build.sh
```
Obviously, when rebuilding, only the last step has to be repeated.

Dependencies may vary between operating systems and distributions.
To build on a clean Ubuntu 24.04 (docker), I required the following:
```sh
RUN apt-get install -y build-essential cmake pkg-config \
    libxcb-glx0-dev libxcb-cursor-dev libxcb-dri2-0-dev \
    libxcb-present-dev libxcb-composite0-dev libxcb-ewmh-dev \
    libxcb-res0-dev \
    libx11-dev libx11-xcb-dev libfontenc-dev libice-dev \
    libsm-dev libxau-dev libxaw7-dev \
    libva-dev libvdpau-dev xkb-data \
    libx11-dev libx11-xcb-dev libfontenc-dev libice-dev libsm-dev \
    libxau-dev libxaw7-dev libxcomposite-dev libxcursor-dev libxdamage-dev \
    libxfixes-dev libxi-dev libxinerama-dev libxkbfile-dev libxmuu-dev \
    libxrandr-dev libxrender-dev libxres-dev libxss-dev libxtst-dev libxv-dev \
    libxxf86vm-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-keysyms1-dev \
    libxcb-xinerama0-dev libxcb-dri3-dev uuid-dev libxcb-dri3-dev \
    libxcb-util-dev
```

### N.B: Installation without Dependency managers

If neither conan nor vcpkg can be used, OpenCV and SymEngine must be installed
on the system.  We have set up cmake not to use conan when the
hostname starts with idun`, in which case the idunbuild.sh`
script can be used for installation.  This has been designed
for the NTNU Idun cluster, but can be tweaked for other systems.

### Sidenote: Running the GUI through docker 

Docker images have been created to build and run the GUI.
It should be possible to build and run them as follows, assuming a Unix like system.

```sh
cd docker-sim && docker build -t dockergui .
docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -u $(id -u):$(id -g) cosmogui
```

## Troubleshooting

### ERROR: Compiler not defined for compiler.libcxx. Please define compiler value first too

This occurs when conan does not correctly set up the default profile. Sometimes the compiler and version fields are missing.

Run:

```
conan profile show default
```
and check (update) these if they are wrong. Check gcc version using gcc --version.
```
conan profile update settings.compiler=gcc default
conan profile update settings.compiler.libcxx=libstdc++11 default
conan profile update settings.compiler.version=<gcc version e.g 11.4> default
```
These are what I used on Ubuntu, but may vary from system to system.

