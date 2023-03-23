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

# Making it Run

## Running from Precompiled Distribution

1.  Make sure you have Python 3.10 installed.
2.  Download and unpack `CosmoSimPy.zip` from 
    [v2.0.2.](https://github.com/CosmoAI-AES/CosmoSim/releases/tag/v2.0.2)
    or
    [the latestrelease](https://github.com/CosmoAI-AES/CosmoSim/releases/).
    Temporary problems with the build on github, may mean that there is no
    precompiled version with the latest release.
3.  Run `CosmoSimPy/CosmoGUI.py` in python.  This is the GUI tool.
4.  The `CosmoSimPy/datagen.py` is the CLI tool and should be run
    on the command line; see below.

## Building from Source

Provided a full C++ installation is available with cmake and conan,
the system is built with

```sh
conan install . -if build
cmake . -B build
cmake --build build
```

This builds the C++ library and the Python library (wrapper).
The [Conan Tutorial](https://docs.conan.io/en/latest/getting_started.html)
recommends the following settings (before building):

```
conan profile new default --detect  # Generates default profile detecting GCC and sets old ABI
conan profile update settings.compiler.libcxx=libstdc++11 default  # Sets libcxx to C++11 ABI
```

There are recurring problems with broken dependencies on conan.  This seems
to be out of our control.  The building scripts suddenly break even with no
change on our side.  

### Dependencies

Using conan, it will tell you about any missing libraries that have to 
be installed system level.  The following commands is what I needed on a
Debian system, and may be good start saving some time.  

```sh
sudo pip3 install conan

sudo apt-get install libgtk2.0-dev libva-dev libx11-xcb-dev libfontenc-dev libxaw7-dev libxkbfile-dev libxmuu-dev libxpm-dev libxres-dev libxtst-dev libxvmc-dev libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev libxcb-util-dev libxcb-util0-dev libvdpau-dev
```

## Installation without Conan

If conan is not available, OpenCV and SymEngine must be installed
on the system.  We have set up cmake not to use conan when the
hostname starts with `idun`, in which case the `idunbuild.sh`
script can be used for installation.  This has been designed
for the NTNU Idun cluster, but can be tweaked for other systems.

## Using Docker 

Docker images have been created to build and run the new python GUI.
It should be possible to build and run them as follows, assuming a Unix like system.

```sh
( cd docker-sim && docker build -t dockersim . )
docker build -t dockergui .
docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -u $(id -u):$(id -g) cosmogui
```

# Running the Software 

There are two tools, a GUI tool and a CLI, discussed below.
First a quick test.

## Test

Once built, an illustrative test set can be generated by
the following command issued from the root directory.

```sh
python3 Python/datagen.py -CR -Z 400 --csvfile Datasets/debug.csv
```

This generates a range of images in the current directory

The flags may be changed; `-C` centres det distorted image in the centre
of the image (being debugged); `-Z` sets the image size; `-R` prints an
axes cross.

## GUI Tool

```sh
python3 Python/CosmoGUI.py` 
```

The GUI tool is hopefully quite self-explanatory.  
The images shown are the actual source on the left and the distorted (lensed)
image on the right.

## Image Generator (CLI)

```sh
Python/datagen.py -S sourcemodel -L lensmodel -x x -y y -s sigma -X chi -E einsteinR -n n -I imageSize -N name -R -C
Python/datagen.py --csvfile Datasets/debug.csv --mask -R -C
Python/datagen.py --help
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

+ `Python/datasetgen.py` makes a CSV file of random parameter sets.
  It should be tweaked to get the desired distribution.
+ `python3 Python/datasetgen.py --help` for instructions


## Scripts

+ `amplitudes.py` generates the 50.txt file, which is used by all
  the libraries to get formulæ for alpha and beta.
+ `montage.sh` concatenates the different images for one source 
  for easier display.
    + This has not been tested with the latest updates.

# Versions

The main branches are

- develop is the current state of the art
- master should be the last stable version

Tags

- v0.1.0, v0.2.0, v1.0.0 are versions made by the u/g students
  Spring 2022.
- v1.0.1 is cleaned up to be able to build v1.0.0
- v2.0.0 (19 Dec 2022) provides the new GUI and CLI tools written
  in Python, with several new features and corrected models.
- v2.0.1..3 patches to v2.0.0

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

# Technical Design

## Components

### C++ components

+ Lens Models
    + `LensModel.cpp` is the abstract base class.
    + `PointMassLens.cpp` simulates the point mass model
      using the exact formulation
    + `RoulettePMLens.cpp` simulates the point mass model using
      the Roulette formalism
    + `SphereLens.cpp` simulates the SIS model
+ Source Models
    + `Source.cpp` is the abstract base class.
    + `SphericalSource.cpp` is standard Guassian model
    + `EllipsoidSource.cpp` is an ellipsoid Guassian model
    + `TriangleSource.cpp` is a three colour triangle source,
       intended for debugging
+ `simaux.cpp` is auxiliary functions
+ `CosmoSim.cpp` defines the `CosmoSimPy` class with python bindigs.
  This class operates as a facade to the library, and does not 
  expose the individual classes.

### Python Components

+ `CosmoSim` is a wrapper around `CosmoSimPy` from `CosmoSim.cpp`,
  defining the `CosmoSim` class.
+ `CosmoGUI` is a tkinter desktop application, providing a GUI to the
  simulator
+ `CosmoSim.View` is a tkinter widget displaying the source and 
  distorted image for `CosmoGUI`.
+ `CosmoSim.Controller` is a tkinter widget to interactively set
  the simulator parameters for `CosmoGUI`.
+ `CosmoSim.Image` provides post-processing functions for the images.
+ `datagen.py` is a batch script to generate distorted images.


## Lens Model Class

### Virtual Functions

The following virtual functions have to be overridden by most subclasses.
They are called from the main update function and overriding them, the entire
lens model changes.

+ `calculateAlphaBeta()`
  pre-calculates $\alpha$ and $\beta$ in the distortion equation.
+ `getDistortedPos()`
  calculates the distortion equation for a give pixel.

The constructor typically has to be overridden as well, to load the formulæ for
$\alpha$ and $\beta$.

### Setters 

Setters are provided for all of the control parameters.

+ `updateXY` to update the $(x,y)$ co-ordinates of the actual image, the
  relative distance $\chi$ to the lens compared to the source, and the
  Einstein radius $R_E$.
  This has to update the apparent position which depends on all of these
  variables.
+ `updateSize` to update the size or standard deviation of the source.
+ `updateNterms` to update the number of terms in the sum after truncation
+ `updateAll` to update all of the above

### Getters

Getters are provided for the three images.

+ `getActual()`
+ `getApparent()`
+ `getDistorted()`

### Update

The main routine of the `Simulator` is `update()` which recalculates the 
three images: actual, apparent, and distorted.  This is called by the setters.

In addition to the virtual functions mentioned above, it depends on

+ `parallelDistort()` and `distort()` which runs the main steps in parallel.
+ `drawParallel()` and `drawSource()` which draws the source image.

## Auxiliaries 

The `simaux.cpp` file provides the following:

+ `factorial_()`
+ `refLines()` to draw the axis cross

### Subclasses

The `Simulator` class implements the point mass model as a default, but the 
subclass `PointMassSimulator` should be used for instantiation.  It overrides
nothing, but the `Simulator` superclass may be made abstract in the future.

The `SphereSimulator` overrides the constructor and the two virtual functions.
The constructor loads the formulæ for `\alpha` and `\beta` which are calculated
by `calculateAlphaBeta()` when parameters change.

# Development and Advanced Testing

## Roulette Test (experimental)

The `roulettetest` program is similar to the old `makeimage`, but can generate extra
images to test the behaviour of the Roulettes model.  In particular,

+ `-A` takes a comma separated list of floating point values, which are taken
  to the the apparent position (distance from origin).  A distorted image is
  created for each value.
+ `-Y` generates a secondary distorted image using the second root for the
  apparent position.  This only makes sense for the point mass lens.

## Regression Testing

The `Scripts/test.sh` script can be used for regression testing.
It generates test images and compare them to images generated by v2.0.2.
If discrepancies are found, it indicates that the simulator has functionaly
changed.


# Contributors

+ **Idea and Current Maintainance** Hans Georg Schaathun <hasc@ntnu.no>
+ **Mathematical Models** Ben David Normann
+ **Initial Prototype** Simon Ingebrigtsen, Sondre Westbø Remøy,
  Einar Leite Austnes, and Simon Nedreberg Runde
