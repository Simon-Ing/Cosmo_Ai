
# CosmoSim Change Log


## [Unreleased] - yyyy-mm-dd

### Added

- Support for sampled lens models using the Roulette formalism
- Test framework to compare output images.

### Changed

- Cleaned up code to make variable names more consistent with 
  mathematical papers

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

