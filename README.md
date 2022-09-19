---
title: The CosmoAI project
---

# Install Dependencies

```sh
sudo pip3 install conan

sudo apt-get install libgtk2.0-dev libva-dev libx11-xcb-dev libfontenc-dev libxaw7-dev libxkbfile-dev libxmuu-dev libxpm-dev libxres-dev libxtst-dev libxvmc-dev libxcb-render-util0-dev libxcb-xkb-dev libxcb-icccm4-dev libxcb-image0-dev libxcb-keysyms1-dev libxcb-randr0-dev libxcb-shape0-dev libxcb-sync-dev libxcb-xfixes0-dev libxcb-xinerama0-dev libxcb-dri3-dev libxcb-util-dev libxcb-util0-dev
```

# Components

Three source files are used

+ `./Simulator.cpp` for the library
+ `./Data_generator.cpp` for the `Datagen` binary. The executable takes four arguments for which I have not found the documentation.
+ `./GL_Simulator_2.cpp` for the GUI Simulator tool, which does run on Debian, although the spherical model seems to be wrong.

There is also code in the `CosmoAI_qt` directory, but this seems not to be used.  The build runs without error even if it is removed.
