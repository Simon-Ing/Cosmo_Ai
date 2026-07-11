# CosmoSim Change Log

## 3.2.2 - 2026-07-11

### Fixed

+ Fixed parameterisation of sersic ellipsoids for the notebook API
+ Fixed bug in GUI with premature garbage collection for sampled lenses.
+ Fixed binding of SIE::getXI to correct namespace
+ Bugfix, binding the drawCaustics function.
+ Fixed failure to load amplitudes file for point mass
+ Fixed exception showing critical curves in GUI
+ Match startup parameters in the GUI display and the simulator. 
+ Fixed failure to run SIE lens in GUI.
+ Fixed segmentation vialation in GUI arising from garbage collection.

## 3.2.1 - 2026-07-09

### Fixed

+ Bugfix, binding the drawCritical function.

## 3.2.0 - 2026-07-09

### Changed

+ Refactored to instantiate objects in python and simplify API.
  Removed the CosmoSim class implemented in C++.
+ Parameterised the Sersic profile for ellipsoid sources and made
  it analogous to the Spherical one.
+ Removed the closed form simulators for point mass lenses, 
  leaving only the modular model.

### Fixed

+ Fixed the memory leak arising from multiple instances of the
  backend simulator.
+ Fixed the definition of offset in the CSV output.

Fixed a bug to allow image generation from a TOML file without a CSV file
to store the parameters. E.g.
```
python -m CosmoSim --rnd --toml dataset.toml 
```

## 3.1.2 - 2026-07-01

## Fixed

+ Possible to run `--rnd` without `--csvfile`

## 3.1.1 - 2026-07-01

Fixed a bug to have dataset work with model/lens specification instead of
config.

## 3.1.0 - 2026-06-30

### Fixed

+ Fixed bug in the lens potential of Point Mass in 
  libamplitudes.py

### Added

+ SimImage.getRoulette() to evaluate roulette amplitudes from python
  in arbitrary precision.
+ Alternative calculation of amplitude formulæ, based on the SIE calculation
  but exploiting the symmetry of circular lenses.

## 3.0.5 - 2026-06-25

Version 3.0 will be the last version using the old amplitudes.py
script which accidentally introduce floating point numbers prematurely
into the roulettes amplitudes.

### Changed

+ CosmoSim.Roulettes uses rational numbers instead of floating point.
  The generated amplitudes formulæ have not been changed.

### Fixed

+ Forcing version numbers, hoping to make it run on MacOS.

## 3.0.4 - 2026-06-24


### Added

+ Amplitudes scripts as part of the package
+ Roulette amplitudes for the pointmass lens, allowing modular roulette simulation.

### Fixed

+ Build on Windows with precompiled distribution on PyPi

## 3.0.3 - 2026-06-19

### Fixed

+ Cluster lens generation

## 3.0.2 - 2026-06-18

### Added

+ Image utility functions

## 3.0.1 - 2026-06-17 

### Added

+ Added relative bound on source r for random dataset generation

### Fixed

+ Fixed inconsistencies to make it work with droulette.

## 3.0.0 - 2026-06-15

### Added

+ Made random dataset generation available as API functions.
    + New function `CosmoSim.dataset.readtoml` to read the config 
+ Random generation of datasets with cluster lenses 
+ Better control of verbosity level, including runtim suppression of debug messages from C++
  modules.

### Changed

+ Interpret images in the lens plane, removing CHI as a parameter. 
+ Unified the scripts into the main module.  All command line features are
  now run as `python -m CosmoSim`, except random dataset generation.

### Fixed

+ Made `--sampled` option optional, defaulting to `None`.
+ Removed analytic but flawed calculation of apparent source position for SIE.
  It now falls back on fix-point iteration which works.

## 2.9.0 - 2026-05-08

### Added

+ Parameterised sersic profile

### Changed

+ Removed redundant CLI arguments

### Fixed

+ Removed analytic but flawed calculation of apparent source position for SIE.
  It now falls back on fix-point iteration which works.

## [2.8.1] - 2026-04-13

### Added

+ The `Parameters` class is now available as `CosmoSim.Parameters`
  (not `CosmoSim.Parameters.Parameters`).
+ reflines option to `getImage()` methods

### Fixed

+ Reduced the amount of diagnostic debug information.
+ Preserve original image in `GenericSim`, by copying the image before
  reference lines are added,

## [2.8.0] - 2026-04-13

### Added

+ `CosmoSim.roulettegen.Resimm()` is now intended to be used as an API feature
  in the same way as `CosmoSim.datagen.SimImage()`.

### Fixed

+ the `row` passed to `CosmoSim.SimImage` can now be a pandas Series where
  the filename is used as row index, or Series name.

## [2.7.0] - 2026-XX-XX

### Added

+ `CosmoSim.datagen.SimImage()` is now intended to be used as an API feature, including
  new API methods to add annotations to the image.

## [2.6.8] - 2026-03-25

### Added

+ `CosmoSim.datagen.makeSingle()` can now be used as a library function to 
  generate images

### Fixed

+ refactored to avoid global parameter `args` in datagen.py

## [2.6.7] - 2026-03-15

### Fixed

+ `datagen` takes image size from TOML file

## [2.6.6] - 2026-03-08

### Fixed

+ Undefined reference in roulettegent is fixed.

## [2.6.5] - 2026-03-05

### Added

- New API for roulette resimulation from other programs.

## [2.6.4] - 2026-03-01

### Added

+ Support for SIE in dataset module.

### Fixed

+ Allow missing amplitudes in roulettegen, using zero as default.
  This is because some amplitudes are always zero and may be
  ommitted from ML output.

## [2.6.3] - 2026-02-23

### Fixed

+ Removed `--lightprofile` option, whose default was overriding other
  settings.

## [2.6.2] - 2026-02-23

### Changed

+ Use continuous uniform distribution in dataset generation, instead of 
  integers.

### Added

+ `datasetgen()` now returns a dataframe and filename is optional.

## [2.6.1] - 2026-02-22

### Added

+ Extended getMSheaders() to take both a minm and maxm parameter 

### Fixed

+ Fixed bug which made every source elliptical and Gaussian. 

## [2.6.0] - 2026-02-19

### Added

+ `CosmoSim.datagen` takes the --mldata option which drops superfluous columns
  from the roulette CSV output.
+ Sersic light profiles for source models

## [2.5.3] - 2025-08-05

### Added

+ PyPI packing for installation with pip.

### Changed

+ Most scripts are moved inside the module, to be run via `python -m`

## [2.5.2] - 2025-01-16

### Fixed

+ Fixed roulette CSV output from datagen.py
+ Fixed bug in PointMass lens.  It now works correctly with Raytrace.
  (Minor visible discrepancy against PointMassExact, probably numerical error.)

## [2.5.1] - 2024-10-27

### Added

+ Source constellations.  Multiple sources can be view through the same
  lens (cluster).  (Not supported in GUI.)

### Changed

+ Refactoring.  Simplified, restructured, and commented the CosmoSim object code.
+ Refactoring.  Removed the RouletteSim class, calling RouletteRegenerator
  directly from python code.
+ Refactoring.  Moved source instantiation to Python.  This reduces coupling and
  simplifies parameter passing when a wider range of source models is supported.

## [2.5.0] - 2024-10-20

### Added

+ Support for SIE
+ Support for clusterlenses with several lenses in the same lens plane
+ Allow masking with explicit radius in `roulettegen.py`
+ New tests to compare and analyse models.
+ Python bindings for lens classes

### Fixed

+ More consistent application of masking
    + Raytrace never masks during computation
    + All models allow postprocessing mask with the same radius
      (except roulette resimulation which requires explicit radius)
+ Refactoring, moving features from superclasses to subclasses to simplify API

### Changed

- `--xireference` is now default.  It can be turned off with `--no-xireference`,
  but only the default is tested.

### Caveats

- The Roulette Regenerator needs debugging and documentation.
- SIE is slow and irresponsive in the GUI, probably due to the computational
  complexity of the distortion function

## [2.4.3] - 2024-05-16

### Fixed

+ removed erroneous factor of one half in the critical curve for SIS
+ fixed caustic for SIS so that it matches SIE when it tends towards SIS.

## [2.4.2] - 2024-05-15 

### Added

+ Draw Caustics for SIE/SIS lenses when critical curves are drawn in GUI
+ Exception handling to avoid unimplemented critical curves crashing
  the simulator in the GUI.
+ Workflows build for python 3.12 on Windows and MacOS

## [2.4.1] - 2024-05-09 

### Added

+ Draw Critical Curves for SIE and SIS lenses
+ New workflows for new OS versions, including MacOS 12 and 14.

### Changed

+ Updated workflows to use Node 20 actions.
+ Added workflows to build on MacOS, including arm64.

## [2.4.0] - 2024-04-18 

### Added

+ Installation instructions using vcpkg
+ SIE lens model

### Changed

+ Use fractional pixels when centring the image (centreImage function).
  The old behaviour, using only integer translation is still supported as
  an option (newbehaviour=False)


## [2.3.1] - 2023-08-03 

### Added

- New source class for demonstration, using a photo of Einstein.
  This is supported in the GUI, but not tested in the CLI.

### Fixed

- removed rotatedsis from workflow regression test as it is no
  longer available.

## [2.3.0] - 2023-08-02

This version is the reference version for the paper submitted
to NIK 2023.

### Added

- Regression Tests for the RouletteRegnerator model.
- To avoid some cropping artifacts, datagen.py can compute the larger images
  which are cropped to the desired size after centring.
- Command line option to generate only a subset of the images from roulettegen.py
- New variant of RouletteRegenerator, using xi as reference point.

### Changed

- Separated Point Mass Models as subclasses of RotatedModel and created a Point Mass
  Lens model to hold the Einstein Radius.
- Removed the superfluous SphereLens class.
- Renamed some classes for consistency.

### Fixed

- In LensModel::distort(), fixed the sign of the polar angle theta
  when Cartesian x=0.  This removes an artifact in images from
  RouletteGenerator.
- Python scripts made more tolerant with missing input.
- Centring no longer rounds to integer pixel positions.

## [2.2.2] - 2023-07-06

### Added

- Non-zero exit codes from the compare script to suggest discrepancies.
  These are used in the regression test script so that it gives a non-zero
  exit code upon regression errors.
- Automated and more flexible regression test workflow.

### Changed

- Reviewed the Dockerfile and merged both images into one, running the GUI.

### Fixed

- Fixed distort() functions which swapped x/y co-ordinates causing errors
  for non-spherical sources.

## [2.2.1] - 2023-07-04

### Added

- New github workflow for regression test.

### Changed

- Simplified workflows using actions.

### Fixed

- Fixed image artifacts caused by losing remaining rows in parallelDistort().
- Bugfixes to make it work on Windows and MacOS.
  This includes making destructors virtual.

## [2.2.0] - 2023-07-03

### Added

- Added options to generate data set with roulette amplitudes from datagen.py
- New lens model and python script allowing the specification of the lens only
  in terms of roulette amplitudes.
- New github workflows, including a release workflow.

### Changed

- Refactored to decouple the simulation model (roulette and ray trace)
  from the Lens model (currently just SIS and a sampled version).
- Removed the unused centreMode code in the C++ library.

### Fixed

- Speeded up image centring.
- Fixed an image artifact caused by drawing light from a fractional pixel
  just outside the boundary of the source image.
  This seems to remove non-deterministic effects as well.


## [2.1.0] - 2023-03-28

### Added

- Support for sampled lens models both using the Roulette formalism
  and calculating pixel for pixel.
- New PsiFunctionLens defining the lens in terms of computational 
  definitions of the lensing potential (psi) and its derivatives.
- New test suite to compare different implementations of 
  similar models, such as eact and roulette point mass,
  sampled and functional SIS, and SIS with and without rotation
- Export of psi and kappa maps for some lens models.
  The python script makes 3D surface plot of these maps.

### Changed

- Cleaned up code to make variable names more consistent with 
  mathematical papers
- Refactoring, using a Lens class separate from the LensModel

### Fixed

- Several unfortunate uses of integers and single-precision floats
  have been changed to double for the sake of precision.
- Fixed centring of image for coloured sources

## [2.0.3] - 2023-03-20

### Added

- Test framework to compare output images.

### Fixed

- Integer division in c_p/c_m in SphereLens changed to floating point. 

## [2.0.2] - 2023-02-22

### Added

- Support for more than 50 terms (nterms>50).

### Changed

- Removed the superfluous variable GAMMA (= einsteinR/2) in C++;
  the variable g in the amplitudes is now equal to einsteinR.

### Fixed

- More comments in the amplitudes.py script (previously Amplitudes_gen.py)

## [2.0.1] - 2023-02-08

### Added

- Github actions to build library for MacOS, Linux, and Windows. 

### Changed

### Fixed

- Polar Co-ordinates in CLI tool.
- Cleaned up build configuration to work on Windows and MacOS.

## [2.0.0] - 2022-12-19

New release prepared for u/g project Spring 2023.
This is the release logged.
Notably, it introduces the CLI and GUI tools implemented in Python.

