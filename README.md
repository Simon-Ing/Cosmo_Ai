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


# Build and Run

Provided a full C++ installation is available with cmake and conan, the system is built
with


```sh
mkdir build && cd build
conan install ..
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Binaries will be located under bin/.

The [Conan Tutorial](https://docs.conan.io/en/latest/getting_started.html)
recommends the following settings (before building):

```
conan profile new default --detect  # Generates default profile detecting GCC and sets old ABI
conan profile update settings.compiler.libcxx=libstdc++11 default  # Sets libcxx to C++11 ABI
```

## Install Dependencies

The following dependencies were required on Debian 11.
The critical part is conan (as well as cmake and make).
Depending on the platform, conan may or may not install other dependencies.
On Debian it just gives an error message explaining what to install.

```sh
sudo pip3 install conan

sudo apt-get install libgtk2.0-dev libva-dev libx11-xcb-dev libfontenc-dev libxaw7-dev libxkbfile-dev libxmuu-dev libxpm-dev libxres-dev libxtst-dev libxvmc-dev libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev libxcb-util-dev libxcb-util0-dev libvdpau-dev
```

# Tools

There are two tools, a GUI tool and a CLI too, plus the deprecated 
`Datagen` tool.

## GUI Tool

The GUI tool should be pretty self explanatory.  
The images shown are the actual source on the left and the distorted (lensed)
image on the right.

## Image Generator 

```sh
bin/makeimage [-S] -x x -y y -s sigma -X chi -E einsteinR -n n -I imageSize -N name
```

+ `-S` uses SphereSimulator instead of the default point mass simulator
+ `x` and `y` are the coordinates of the actual source
+ `s` is the standard deviation of the source
+ `chi` is the distance to the lens in percent of the distance to the source
+ `einsteinR` is the Einstein radius of the lens
+ `n` is the number of terms to use in roulette sum.  
  (Not used for the point mass model.)
+ `imageSize` size of output image in pixels.  The image will be
  `imageSize`$\times$`imageSize` pixels.
+ `name` is the name of the simulation, and used to generate filenames.

To bulk generate images, two scripts have been provided:

+ `Scripts/datasetgen.py` makes a CSV file of random parameter sets.
  It should be tweaked to get the desired distribution.
+ `Scripts/datagen.sh` to read the CSV file and generate the corresponding
  images

# Versions

The main branches are

- develop is the current state of the art
- master should be the last stable version
- release/0.1 is the original version, with simulator and GUI in the same class.

The `CosmoAI_qt` directory is unused, and removing it, the code still works.
Yet it may prove a useful starting point for a better GUI based on QT.

# Components

+ Lens Models
    + `LensModel.cpp` is the abstract base class.
    + `PointMassLens.cpp` simulates the point mass model
    + `SphereLens.cpp` simulates the SIS model
+ Source Models
    + `Source.cpp` is the abstract base class.
    + `SphericalSource.cpp` is standard Guassian model
    + `EllipsoidSource.cpp` is an ellipsoid Guassian model
+ `Window.cpp` is the GUI window.
+ Binaries
    + `Simulator.cpp` for the GUI Simulator tool, which does run on Debian, although the spherical model seems to be wrong.
    + `makeimage.cpp` is the CLI Tool
    + `Data_generator.cpp` *(deprecated)* for the `Datagen` binary.

# Technical Design

## Window Class

The `initGui` method sets up a window using the basic GUI functions from OpenCV.
This is crude, partly because the OpenCV API is only really meant for debugging, and
partly because some features are only available with QT and I have not yet figured out
how to set up OpenCV with QT support.

When the GUI is initialised, `initSimulator` is called to instantiate a `Simulator`.

When a trackbar changes in the GUI, the corresponding callback function is called;
either `updateXY`, `updateMode`, `updateSize`, or `updateNterms`.  Except for `updateMode`,
each of these
call corresponding update functions in the simulator before they call
`drawImages()` to get updated images from the simulator and display them.
When the mode changes, `updateMode` will instantiate the relevant subclass of `Simulator`.

The instance variables correspond to the trackbars in the GUI, except for `size` which
gives the image size.  This is constant, currently at 300, and has to be the same for
the `Window` and `Simulator` objects.

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

The main routine of the `Simulator` is `update()` which recalculates the three images,
actual, apparent, and distorted.  This is called by the setters.

In addition to the virtual functions mentioned above, it depends on

+ `parallelDistort()` and `distort()` which runs the main steps in parallel.
+ `drawParallel()` and `drawSource()` which draws the source image.
  Only a Gaussian image has been implemented.  When new source models are required,
  this should be delegated to a separate class.

## Auxiliaries

+ `factorial_()`
+ `writeToPngFiles()` should be moved to the CLI tool

### Subclasses

The `Simulator` class implements the point mass model as a default, but the 
subclass `PointMassSimulator` should be used for instantiation.  It overrides
nothing, but the `Simulator` superclass may be made abstract in the future.

The `SphereSimulator` overrides the constructor and the two virtual functions.
The constructor loads the formulæ for `\alpha` and `\beta` which are calculated
by `calculateAlphaBeta()` when parameters change.

# Contributors

+ **Idea and Current Maintainer** Hans Georg Schaathun <hasc@ntnu.no>
+ **Mathematical Models** Ben David Normann
+ **Initial Prototype** Simon Ingebrigtsen, Sondre Westbø Remøy,
  Einar Leite Austnes, and Simon Nedreberg Runde
