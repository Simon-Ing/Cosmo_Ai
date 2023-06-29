
# CosmoSim Change Log


## [2.2.0] - ?? Unreleased

### Added

- Added options to generate data set with roulette amplitudes from datagen.py
- New lens model and python script allowing the specification of the lens only
  in terms of roulette amplitudes.

### Changed

- Refactored to decouple the simulation model (roulette and ray trace)
  from the Lens model (currently just SIS and a sampled version).
- Removed the unused centreMode code in the C++ library.

### Fixed

- Speeded up image centring.


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

