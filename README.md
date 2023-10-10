---
title: The CosmoAI project
---

This project provides a simulator for gravitational lensing based on
Chris Clarkson's Roulettes framework.
The initial prototype was an undergraduate
[final year project](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3003634)
by Ingebrigtsen, Remøy, Westbø, Nedreberg, and Austnes (2022).
The software includes both a GUI simulator for interactive experimentation, 
and a command line interface for batch generation of datasets.

The intention is to be able to use the synthetic datasets to train
machine learning models which can in turn be used to map the dark
matter of the Universe.  This is work in progress.

Documentation is being written at 
[https://cosmoai-aes.github.io/](https://cosmoai-aes.github.io/),
but it still incomplete and fragmented.

# Making it Run

## Running from Precompiled Version in Python

The libraries are built using Github Workflows and packed
as complete Python modules with lirbraries or scripts.
Currently, we build for Python 3.9, 3,10, and 3.11 on Linux,
and for Python 3.10 and 11 on Windows.
MacOS binaries have to be built manually, and may therefore
be less up to date.
The GUI has not been tested on Windows.

1.  Make sure you have one of the supported Python versions
2.  Download and unpack `CosmoSimPy.zip` from 
    [the latest release](https://github.com/CosmoAI-AES/CosmoSim/releases/latest).
    If a MacOS version exists, it is named as such; there is one for 
    [v2.2.2](https://github.com/CosmoAI-AES/CosmoSim/releases/tag/v2.2.2).
3.  Run `CosmoSimPy/CosmoGUI.py` in python.  This is the GUI tool.
4.  The `CosmoSimPy/datagen.py` is the CLI tool and should be run
    on the command line; see below.

The binaries are not signed, and on MacOS you will have to confirm
that you trust the binary before it will run.

NOTE:
The steps under "Running the Software" do not currently work from the precompiled version. TODO: Build a more complete and rubust release artifact.

# Building from Source

The build procedure is primarily developed on Debian Bullseye, but it now 
also works reliable on github runners running Windows-2019, Ubuntu-20.04,
or Ubuntu 22.04.  We also have it working on MacOS, but we also have problems
with other macs, depending on their setup.  We do not have capacity to develop
generic and robust build procedures, but we shall be happy to incorporate 
contributions.

## Pre-step: Dependencies

Using conan, it will tell you about any missing libraries that have to 
be installed on the system level.  The following commands is what I needed on a Debian system, and may be good start saving some time. 

```sh
sudo apt-get install libgtk2.0-dev libva-dev libx11-xcb-dev libfontenc-dev libxaw7-dev libxkbfile-dev libxmuu-dev libxpm-dev libxres-dev libxtst-dev libxvmc-dev libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev libxcb-util-dev libxcb-util0-dev libvdpau-dev
```

## **Step 1: Install Conan**

The build stack uses conan for dependencies.  It needs to be installed,
in version 1.59, and configured to use the C++11 ABI.
(See [Conan Tutorial](https://docs.conan.io/en/latest/getting_started.html)
for further information.)

```
pip3 install conan==1.59
conan profile new default --detect 
```

Find your gcc version. This is needed for below.

```
gcc --version
```

In ~/.conan/settings.yml, scroll down to the gcc section and make sure your version is in the list of versions. If not, add it there.

```
nano ~/.conan/settings.yml

# Check this section, and make sure the output from gcc --version is in the list. If not, add it.

    gcc: &gcc
        version: ["4.1", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9",
                  "5", "5.1", "5.2", "5.3", "5.4", "5.5",
                  "6", "6.1", "6.2", "6.3", "6.4", "6.5",
                  "7", "7.1", "7.2", "7.3", "7.4", "7.5",
                  "8", "8.1", "8.2", "8.3", "8.4", "8.5",
                  "9", "9.1", "9.2", "9.3", "9.4", "9.5",
                  "10", "10.1", "10.2", "10.3", "10.4",
                  "11", "11.1", "11.2", "11.3", "11.4",
                  "12", "12.1", "12.2"]
```
Now we need to update the conan default profile.
First, run:

```
conan profile show default
```
and compare it to this:
```
[settings]
os=Linux
os_build=Linux
arch=x86_64
arch_build=x86_64
build_type=Release
compiler=gcc
compiler.libcxx=libstdc++11
compiler.version=11.4
[options]
[conf]
[build_requires]
[env]
CC=/usr/bin/gcc
CXX=/usr/bin/g++
```
You will need to update the settings if yours is considerably emptier. Most likely these parameters:

```
conan profile update settings.compiler=gcc default
conan profile update settings.compiler.libcxx=libstdc++11 default
conan profile update settings.compiler.version=<gcc-version> default

conan profile update env.CC=<path/to/gcc>
conan profile update env.CXX=<path/to/g++>
```

You should check your location of gcc and g++. 'usr/bin/' is probably a good bet. E.g I use usr/bin/gcc and usr/bin/g++ above.

## **Step 2: Build**

*NOTE: There are recurring problems with broken dependencies on conan.  This seems to be out of our control.  The building scripts suddenly break even with no change on our side. For instance, it may be necessary explicitly to build OpenCV:*

```sh
# if install command below fails, try:
conan install . -if build --build=missing
```

To build the C++ library and the Python library (wrapper), we use cmake as follows.
```sh
cd {/path/to/CosmoSim/root/dir}
conan install . -if build
cmake . -B build
cmake --build build
```
.

# Installation without Conan

If conan is not available, OpenCV and SymEngine must be installed
on the system.  We have set up cmake not to use conan when the
hostname starts with `idun`, in which case the `idunbuild.sh`
script can be used for installation.  This has been designed
for the NTNU Idun cluster, but can be tweaked for other systems.

# Using Docker 

Docker images have been created to build and run the new python GUI.
It should be possible to build and run them as follows, assuming a Unix like system.

```sh
cd docker-sim && docker build -t dockergui .
docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -u $(id -u):$(id -g) cosmogui
```

# Running the Software 

There are two tools, a GUI tool and a CLI, discussed below.
First a quick test.

## Test

Once built, an illustrative test set can be generated by
the following command issued from the root directory.

```sh
python3 CosmoSimPy/datagen.py -CR -Z 400 --csvfile Datasets/debug.csv
```

This generates a range of images in the current directory

The flags may be changed; `-C` centres det distorted image in the centre
of the image (being debugged); `-Z` sets the image size; `-R` prints an
axes cross.

## GUI Tool

```sh
python3 CosmoSimPy/CosmoGUI.py` 
```

The GUI tool is hopefully quite self-explanatory.  
The images shown are the actual source on the left and the distorted (lensed)
image on the right.

## Image Generator (CLI)

```sh
CosmoSimPy/datagen.py -S sourcemodel -L lensmodel -x x -y y -s sigma -X chi -E einsteinR -n n -I imageSize -N name -R -C
CosmoSimPy/datagen.py --csvfile Datasets/debug.csv --mask -R -C
CosmoSimPy/datagen.py --help
```

The second form generates images in bulk by parsing the CSV file.
Parameters which are constant for all images may be given on the 
command line instead of the CSV file.

+ `lensmodel` is `p` for point mass (exact), `r` for Roulette (point mass),
  or `s` for SIS (Roulette).
+ `sourcemodel` is `s` for sphere, `e` for ellipse, or `t` for
   triangle.
+ `-C` centres the image on the centre of mass (centre of light)
+ `-R` draw the axes cross
+ `x` and `y` are the coordinates of the actual source
+ `s` is the standard deviation of the source
+ `chi` is the distance to the lens in percent of the distance to the source
+ `einsteinR` is the Einstein radius of the lens
+ `n` is the number of terms to use in roulette sum.  
  (Not used for the point mass model.)
+ `imageSize` size of output image in pixels.  The image will be
  `imageSize`$\times$`imageSize` pixels.
+ `name` is the name of the simulation, and used to generate filenames.
+ `--help` for a complete list of options.

To bulk generate images the following script creates a CSV file
to use with the `--csvfile` option above.

+ `CosmoSimPy/datasetgen.py` makes a CSV file of random parameter sets.
  It should be tweaked to get the desired distribution.
+ `python3 CosmoSimPy/datasetgen.py --help` for instructions


## Scripts

+ `amplitudes.py` generates the 50.txt file, which is used by all
  the libraries to get formulæ for alpha and beta.
+ `montage.sh` concatenates the different images for one source 
  for easier display.
    + This has not been tested with the latest updates.

# Use cases

## Training sets for roulette amplitudes

The datasets generated from `datasetgen.py` give the parameters for the 
lens and the source, as well as the image file.
This allows us to train a machine learning model to identify the lens
parameters, *assuming* a relatively simple lens model.
It is still a long way to go to map cluster lenses.

An alternative approach is to try to estimate the effect (lens potential)
in a neighbourhood around a point in the image.  For instance, we may want
to estimate the roulette amplitudes in the centre of the image.
The `datagen.py` script can generate a CSV file containing these data along
with the image, as follows:

```sh
mkdir images
python3 CosmoSimPy/datagen.py -C -Z 400 --csvfile Datasets/debug.csv \
        --directory images --outfile images.csv --nterms 5
```

The images should be centred (`-C`); the amplitudes may not be
meaningful otherwise.  The `--directory` flag puts images in
the given directory which must exist.  The image size is given by
`-Z` and is square.  The input and output files go without saying.
The number of terms (`--nterms`) is the maximum $m$ for which the
amplitudes are generated; 5 should give about 24 scalar values.

The amplitudes are labeled `alpha[`$s$,$m$`]` and `beta[`$s$,$m$`]` 
in the outout CSV file.  One should focus on predicting the amplitudes
for low values of $m$ first.  The file also reproduces the source
parameters, and the centre of mass $(x,y)$ in the original co-ordinate
system using image coordinates with the origin in the upper left corner.

The most interesting lens model for this exercise is PsiFunctionSIS (fs),
which gives the most accurate computations.  The roulette amplitudes have
not been implemented for any of the point mass lenses yet, and it also
does not work for «SIS (rotated)» which is a legacy implementation of
the roulette model with SIS and functionally equivalent to «Roulette SIS«
(rs).

**Warning** This has yet to be tested properly.

# Versions

The main branches are

- develop is the current state of the art
- master should be the last stable version

## Releases

- v-test-* are test releases, used to debug workflows.  Please ignore.
- v2.0.0 (19 Dec 2022) provides the new GUI and CLI tools written
  in Python, with several new features and corrected models.
    - v2.0.1..3 patches to v2.0.0

## Older Tags

Prior to v2.0.0 some releases have been tagged, but not registered
as releases in github.

- v0.1.0, v0.2.0, v1.0.0 are versions made by the u/g students
  Spring 2022.
- v1.0.1 is cleaned up to be able to build v1.0.0

# Caveats

The simulator makes numerical calculations and there will always
be approximation errors.

1.  The images generated from the same parameters have changed slightly
    between version.  Some changes are because some unfortunate uses of
    integers and single-precision numbers have been avoided, and some 
    simply because the order of calculation has changed.
1.  The SIS model is implemented in two versions, one rotating
    to have the source on the x-axis and one working directly with
    arbitrary position.  Difference are not perceptible by visual
    comparison, but the difference image shows noticeable difference.


# Contributors

+ **Idea and Current Maintainance** Hans Georg Schaathun <hasc@ntnu.no>
+ **Mathematical Models** Ben David Normann
+ **Initial Prototype** Simon Ingebrigtsen, Sondre Westbø Remøy,
  Einar Leite Austnes, and Simon Nedreberg Runde
