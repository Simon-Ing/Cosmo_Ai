---
title: The CosmoAI project
---

# Build and Run

Provided a full C++ installation is available with cmake and conan, the system is built
with

```sh
cmake -DCMAKE_BUILD_TYPE=Release .
make
```

Binaries will be located under bin/.

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

There are two tools, a GUI tool and a CLI tool

## GUI Tool

The GUI tool should be pretty self explanatory.  
The images shown are the actual source on the left and the distorted (lensed)
image on the right.

## CLI Tool

This can be run like this:

```sh
bin/Datagen 20 512 test 50
```
+ Generate 20 images (first parameters)
+ Image size is $512\times512$
+ Images are place under the directory `test/images` and `test/actual`.
  **Note** The directories must be exist.
+ The distance to the lens is half (50%) of the distance to the source.
  If the last parameter is 0, the distance is drawn at random.

# Versions

There are two branches.

- release/0.1 is the original version, with simulator and GUI in the same class.
- master/refactor-separate-io which separates the GUI from the simulator 

The original also had code in the `CosmoAI_qt` directory.  
It is unused, and removing it, the code still works.
Yet it may prove a useful starting point for a better GUI based on QT.

The master branch is retained as a reference implementation. 

## Master Branch Components

Three source files are used

+ `./Simulator.cpp` for the library
+ `./Data_generator.cpp` for the `Datagen` binary. The executable takes four arguments for which I have not found the documentation.
+ `./GL_Simulator_2.cpp` for the GUI Simulator tool, which does run on Debian, although the spherical model seems to be wrong.


## Refactored Components

+ Simulators
    + `Simulator.cpp` is the base class; in practice it also contains the code for the point mass model,
      but this may change.
    + `PointMassSimulator.cpp` simulates the point mass model
    + `SphereSimulator.cpp` simulates the SIS model
+ `Window.cpp` is the GUI
+ `Data_generator.cpp` for the `Datagen` binary. The executable takes four arguments for which I have not found the documentation.
+ `GL_Simulator_2.cpp` for the GUI Simulator tool, which does run on Debian, although the spherical model seems to be wrong.

The data generator has not been refactored and does not work at present.
The code for file output has been retained in the simulator class, but will be moved when
the data generator is repaired.

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

## Simulator Class

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
