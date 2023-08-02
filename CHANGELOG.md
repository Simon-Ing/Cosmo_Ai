
# CosmoSim Change Log

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

