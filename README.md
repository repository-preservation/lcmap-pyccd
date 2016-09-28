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

[Test Data](docs/TestData.md)
[Reference Implementation](https://github.com/USGS-EROS/matlab-ccdc/blob/master/TrendSeasonalFit_v12_30ARDLine.m)
