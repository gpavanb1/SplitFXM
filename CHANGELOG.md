# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.2] - 2024-10-08
## Fixed
- Downgraded `numpy` to `1.26.4` for broaded compatibility instead of `2`

## [0.4.2] - 2024-10-08
## Added
- Conservative Sod Shock Tube tutorial with verification

## Fixed
- Added `MANIFEST.in` to include Cython files

## [0.4.1] - 2024-10-06
## Added
- Sod Shock Tube tutorial

## Fixed
- Typecasting issue in Euler timestepping

## [0.4.0] - 2024-10-02
## Added
- Cython-based Finite Difference Methods and Finite Volume Methods
- Vector Finite Volume
    - Support for flux Jacobian for all schemes
    - Characteristic-variable based approach for all schemes. Handles full matrix flux Jacobians
- Scheme options - for usage such as limiters

## Fixed
- Single-equation models only
- Removed lists in favour of multidimensional NumPy arrays
- `evolve` now advances till `t_diff` is reached for Euler
- `test_validation` renamed to `test_verification`
- Prompt in an FVM scheme error message

## [0.3.4] - 2024-09-18
## Fixed
- Added `max_step` to time-integration
- Fixed `evolve` to use `max_step` and validation test

## [0.3.3] - 2024-09-17
## Added
- Added various FVM schemes along with tests

## [0.3.2] - 2024-09-16
## Fixed
- License terms in `setup.py` and license file rendering
- Correction to simulation docs regarding Jacobian-based time integration


## [0.3.1] - 2024-09-16
## Added
- Scipy-based time integration with Euler support

## [0.3.0] - 2024-09-15
## Added
- PyTest with ~100% coverage
- Compact scheme generation for uneven grid spacing
- Asymmetric finite difference scheme example and benchmark
- Updated benchmark to compare for both symmetric and asymmetric stencil
- Pricing model and updated license

## Fixed
- Outflow BC for right side
- Remove unreachable code in `bc.py`
- Dictionary-type BCs that raises on incorrect data
- Stencil size checks with domain


## [0.2.0] - 2024-08-29
## Added
- Added support for asymmetric stencils
- Documentation (not in PyPI)


## [0.1.0] - 2024-08-29
### Added
- Initial release with core functionality
- Sparse Jacobian Benchmark

### Fixed
- Dense and Sparse Jacobian matching with `np.allclose` for various boundary conditions


