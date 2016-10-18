# PyCCD - Python Continuous Change Detection
pyccd exists to provide the simplest possible implementation of ccd.

## Using PyCCD
```python
>>> import ccd
>>> results = ccd.detect(dates, reds, greens, blues, nirs, swir1s, swir2s, thermals, qas)
>>>
>>> type(results)
<class 'tuple'>
>>>
>>> type(results[0])
<class 'dict'>
>>>
>>> results
(
{algorithm:'pyccd:x.x.x'
 start_day:int,
 end_day:int,
 observation_count:int,
 red:      {magnitude:float,
            rmse:float,
            coefficients:(float, float, ...),
            intercept:float},
 green:    {magnitude:float,
            rmse:float,
            coefficients:(float, float, ...),
            intercept:float},
 blue:     {magnitude:float,
            rmse:float,
            coefficients:(float, float, ...),
            intercept:float},
 nir:      {magnitude:float,
            rmse:float,
            coefficients:(float, float, ...),
            intercept:float},
swir1:    {magnitude:float,
           rmse:float,
           coefficients:(float, float, ...),
           intercept:float},
swir2:    {magnitude:float,
           rmse:float,
           coefficients:(float, float, ...),
           intercept:float}
},
)
```

## Installing
System requirements (Ubuntu)
* python3-dev
* gfortran
* libopenblas-dev
* liblapack-dev
* graphviz
* python-virtualenv

System requirements (Centos)
* python3-devel
* gfortran
* blas-dev
* lapack-dev
* graphviz
* python-virtualenv

It's highly recommended to do all your development & testing in a virtual environment.
```bash
user@dev:/home/user/$ mkdir pyccd
user@dev:/home/user/$ cd pyccd
user@dev:/home/user/pyccd$ virtualenv -p python3 .venv
user@dev:/home/user/pyccd$ . .venv/bin/activate
(.venv) user@dev:/home/user/pyccd$
```

The rest of the command prompts are truncated to ```$``` for readability, but assume an activated virtual environment and pwd as above, or that you know what you are doing.

##### Clone the repo
```bash
$ git clone https://github.com/usgs-eros/lcmap-pyccd.git
<<<<<<< HEAD
=======
```
or if you have ssh keys set up in github:
```bash
$ git clone git@github.com:usgs-eros/lcmap-pyccd.git
>>>>>>> corrected installation failures from setup.py attempting to import __init__.py prior to dependencies being installed
```
##### Install test dependencies
```bash
$ pip install -e .[test]
```

## Testing & Running
```bash
$ pytest
$ pytest --profile
$ pytest --profile-svg

# pytest-watch
$ ptw
```

##### Running via command-line
```bash
$ python ./ccd/cli.py sample test/resources/sample_2.csv
```

## Contributing
Contributions to pyccd are most welcome, just be sure to thoroughly review the guidelines first.

[Contributing](docs/CONTRIBUTING.md)

[Developers Guide](docs/DEVELOPING.md)

## Versions
PyCCD versions comply with [PEP440](https://www.python.org/dev/peps/pep-0440/)
and [Semantic Versioning](http://semver.org/), thus MAJOR.MINOR.PATCH.LABEL as
defined by:

> Given a version number MAJOR.MINOR.PATCH, increment the:

> 1. MAJOR version when you make incompatible API changes

> 2. MINOR version when you add functionality in a backwards-compatible manner, and

> 3. PATCH version when you make backwards-compatible bug fixes.

> Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

Alpha releases (x.x.x.ax) indicate that the code functions but the result may
or may not be correct.

Beta releases (x.x.x.bx) indicate that the code functions and the results
are believed to be correct by the developers but have not yet been verified.

Release candidates (x.x.x.rcx) indicate that the code functions and the results
are correct according to the developers and verifiers and is ready for final
performance and acceptance testing.

Full version releases (x.x.x) indicate that the code functions, the results
are verified to be correct and it has passed all testing and quality checks.

PyCCD's version is defined by the ```ccd/version.py/__version__``` attribute
ONLY.

## References

### ATBD
1. Obtain minimum number of clear observations
2. Run regression against this set (n)
3. Continue adding observations
4. If three additional consecutive observations falls outside predicted
   values for any band, a change has occurred for all bands
   and a new regression model is started.
5. If next three observations are not outside the range, an outlier has
    has been detected.
* Outliers are flagged and omitted from the regression fitting

Links
* [Test Data](docs/TestData.md)
* [Reference Implementation](https://github.com/USGS-EROS/matlab-ccdc/blob/master/TrendSeasonalFit_v12_30ARDLine.m)
* [Landsat Band Specifications](http://landsat.usgs.gov/band_designations_landsat_satellites.php)
* [Landsat 8 Surface Reflectance Specs](http://landsat.usgs.gov/documents/provisional_lasrc_product_guide.pdf)
* [Landsat 4-7 Surface Reflectance Specs](http://landsat.usgs.gov/documents/cdr_sr_product_guide.pdf)
