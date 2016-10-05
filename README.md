# PyCCD - Python Continuous Change Detection
pyccd exists to provide the simplest possible implementation of ccd.

## Using PyCCD
```python
>>> import ccd
>>> results = ccd.detect(dates, reds, greens, blues, nirs, swir1s, swir2s, thermals, qas)
>>> 
>>> type(results)
tuple
>>>
>>> type(results[0])
collections.namedtuple
>>>
>>> results[0]
(
            (start_time=int, end_time=int, observation_count=int,
             red =     (magnitudes=float,
                        rmse=float,
                        coefficients=(float, float, ...),
                        intercept=float),
             green =   (magnitudes=float,
                        rmse=float,
                        coefficients=(float, float, ...),
                        intercept=float),
             blue =    (magnitudes=float,
                        rmse=float,
                        coefficients=(float, float, ...),
                        intercept=float),
             nir =     (magnitudes=float,
                        rmse=float,
                        coefficients=(float, float, ...),
                        intercept=float),
             swir1 =   (magnitudes=float,
                        rmse=float,
                        coefficients=(float, float, ...),
                        intercept=float),
             swir2 =   (magnitudes=float,
                        rmse=float,
                        coefficients=(float, float, ...),
                        intercept=float),
             thermal = (magnitudes=float,
                        rmse=float,
                        coefficients=(float, float, ...),
                        intercept=float),
            ),
        )
```

## Installing
System requirements for PyCCD
* python3-dev (ubuntu) or python3-devel (centos)
* gfortran
* libopenblas-dev
* liblapack-dev
* graphviz

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
$ git clone https://github.com/davidvhill/pyccd.git
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
```

##### Running via command-line
```bash
$ python ./ccd/cli.py
$ pyccd
```

## Contributing
Contributions to pyccd are most welcome, just be sure to thoroughly review the guidelines first.

[Contributing](docs/CONTRIBUTING.md)

[Developers Guide](docs/DEVELOPING.md)


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
