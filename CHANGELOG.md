# pyccd change log
All notable changes to this project will be documented in this file. Changes before 1.0.0.b1 are not tracked.  

## 1.4.0 - 2017-4-26
### Added
 - this file
 - can now pass a processing parameter dictionary to ccd.detect, key/values will override parameters.yaml
 - support for ARD bit-packed QA values
 
### Changed
 - overhauled code base to have some separation of paradigms, procedures.py relates to the more procedural code while changes.py should be more functional in nature
 - tried to remove all default argument values for all methods
 - adjusted many internal method signatures
 - removed ensure_ndarray_input decorator
 - logging is less global and listens to what imports ccd
 
### Bug Fixes
 - reference period array rather than original dates array for the model window in the initialize method
 - simplified logic and removed find_time_index method
 - changed adjusted variogram logic to match the Matlab approach 1:1

## [1.3.1] - 2017-4-10
### Changed
 - move duplicate date trimming to occur after other masking is done
 - ensure_ndarray_input to use keyword args better [broken]
 
### Bug Fixes
 - removed errant print statement
 - fixed the adjusted variogram calculation introduced last version

## [1.3.0] - 2017-3-30
### Changed
 - robust iterative reweighted least squares for the regression used by the partial Tmask, alignment with original Matlab CCDC v12.30
 - updated variogram calculation to try and only use observations > 30 days a part
 - TestData.md now copy/pastable

## [1.1.0] - 2017-3-2
### Added
 - auto-deploy to pypi from master
 
### Bug Fixes
 - per band change magnitudes based on the final peek window of a time segment

## [1.0.4.b1] - 2017-02-24
### Added
 - travis-ci automated testing and deployment setup
 - better pypi interactions
 
### Changed
 - time segment QA values aligned with original Matlab CCDC v12.30
 - general fits removed between stable segments
 
### Bug Fixes
 - updated setup.py for numpy >= 1.10

## [1.0.0.b1] - 2017-01-31
 - Initial beta release for evaluation efforts
 - Change tracking start

## 1.0.0.a1 - 2016-10-13
 - Proof of concept for moving the CCDC code base to python

[1.0.0.b1]: https://github.com/usgs-eros/lcmap-pyccd/compare/1.0.0.a1...1.0.0.b1
[1.0.4.b1]: https://github.com/usgs-eros/lcmap-pyccd/compare/1.0.0.b1...v1.0.4.b1
[1.1.0]: https://github.com/usgs-eros/lcmap-pyccd/compare/v1.0.4.b1...v1.1.0
[1.3.0]: https://github.com/usgs-eros/lcmap-pyccd/compare/v1.1.0...v1.3.0
[1.3.1]: https://github.com/usgs-eros/lcmap-pyccd/compare/v1.3.0...v1.3.1
[1.4.0]: https://github.com/usgs-eros/lcmap-pyccd/compare/v1.3.1...v1.4.0