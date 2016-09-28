## ATBD
1. Obtain minimum number of clear observations
2. Run regression against this set (n)
3. Continue adding observations
4. If three additional consecutive observations falls outside predicted
   values for any band, a change has occurred for all bands
   and a new regression model is started.
5. If next three observations are not outside the range, an outlier has
    has been detected.

* Outliers are flagged and omitted from the regression fitting

## Purpose
pyccd exists to provide the simplest possible implementation of ccd.

## Performance TODO
* optimize data structures (numpy)
* use pypy
* employ @lrucache

## Developing
It's highly recommended to create a virtual environment to perform all
your development.
```bash
$ cd pyccd
$ virtualenv -p python3 .
```

Pull down this code.
```bash
$ cd pyccd
$ git clone https://github.com/davidvhill/pyccd.py
```

## Testing
```bash
$ python setup.py test
```
## References

### [Test Data](docs/TestData.md)

### [Reference Implementation](https://github.com/USGS-EROS/matlab-ccdc/blob/master/TrendSeasonalFit_v12_30ARDLine.m)

### [Landsat Band Specifications](http://landsat.usgs.gov/band_designations_landsat_satellites.php)

### [Landsat 8 Surface Reflectance Specs](http://landsat.usgs.gov/documents/provisional_lasrc_product_guide.pdf)

### [Landsat 4-7 Surface Reflectance Specs](http://landsat.usgs.gov/documents/cdr_sr_product_guide.pdf)
