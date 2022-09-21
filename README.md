---
title: The CosmoAI project
---

# Install Dependencies

```sh
sudo pip3 install conan

sudo apt-get install libgtk2.0-dev libva-dev libx11-xcb-dev libfontenc-dev libxaw7-dev libxkbfile-dev libxmuu-dev libxpm-dev libxres-dev libxtst-dev libxvmc-dev libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev libxcb-util-dev libxcb-util0-dev
```

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

