# PyCCD - Python Continuous Change Detection
pyccd exists to provide the simplest possible implementation of ccd.

## Using PyCCD
```python
>>> import ccd
>>> results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas)
>>>
>>> type(results)
<class 'dict'>
>>>
>>> results
{algorithm: 'pyccd:x.x.x',
 processing_mask: [bool, bool, ...],
 snow_prob: float,
 water_prob: float,
 cloud_prob: float,
 change_models: [
     {start_day: int,
      end_day: int,
      break_day: int,
      observation_count: int,
      change_probability: float,
      curve_qa: int,
      blue:      {magnitude: float,
                  rmse: float,
                  coefficients: (float, float, ...),
                  intercept: float},
      green:     {magnitude: float,
                  rmse: float,
                  coefficients: (float, float, ...),
                  intercept: float},
      red:       {magnitude: float,
                  rmse: float,
                  coefficients: (float, float, ...),
                  intercept: float},
      nir:       {magnitude: float,
                  rmse: float,
                  coefficients: (float, float, ...),
                  intercept: float},
      swir1:     {magnitude: float,
                  rmse: float,
                  coefficients: (float, float, ...),
                  intercept: float},
      swir2:     {magnitude: float,
                  rmse: float,
                  coefficients: (float, float, ...),
                  intercept: float},
      thermal:   {magnitude: float,
                  rmse: float,
                  coefficients: (float, float, ...),
                  intercept: float}}
                 ]
}

```

Default processing parameters can be over-ridden using a dictionary (see parameters.yaml for valid keys):  

```python
>>> import ccd
>>> params = {'QA_BITPACKED': False,
              'QA_FILL': 255,
              'QA_CLEAR': 0,
              'QA_WATER': 1,
              'QA_SHADOW': 2,
              'QA_SNOW': 3,
              'QA_CLOUD': 4}
>>> results = ccd.detect(dates, blues, greens, reds, nirs, swir1s, swir2s, thermals, qas, params=params)
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
```
or if you have ssh keys set up in github:
```bash
$ git clone git@github.com:usgs-eros/lcmap-pyccd.git
```

##### Install dev dependencies
Install jupyter notebook and line_profiler
```bash
$ pip install -e .[dev]
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

## Profiling
Decorate the function to be profiled with ```@profile``` and
run ```make profile```.  Remove decorations before committing code.


## Contributing
Contributions to pyccd are most welcome, just be sure to thoroughly review the guidelines first.

[Contributing](docs/CONTRIBUTING.md)

[Developers Guide](docs/DEVELOPING.md)

## Versions
PyCCD previously followed MAJOR.MINOR.PATCH.LABEL semantic versioning but has
changed to date based semantic versioning, thus: YYYY.MM.DD[.HH.MM.SS][-label].

PyCCD's version is defined by the ```ccd/version.py/__version__``` attribute
ONLY.

See [Semantic Versioning](http://semver.org/).
## References

Links
* [Test Data](docs/TestData.md)
* [Reference Implementation](https://github.com/USGS-EROS/matlab-ccdc/blob/master/TrendSeasonalFit_v12_30ARDLine.m)
* [Landsat Band Specifications](http://landsat.usgs.gov/band_designations_landsat_satellites.php)
* [Landsat 8 Surface Reflectance Specs](http://landsat.usgs.gov/documents/provisional_lasrc_product_guide.pdf)
* [Landsat 4-7 Surface Reflectance Specs](http://landsat.usgs.gov/documents/cdr_sr_product_guide.pdf)
