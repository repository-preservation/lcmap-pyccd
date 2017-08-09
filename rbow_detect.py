from datetime import datetime
from test.shared import snap, rainbow
import numpy as np
import pandas as pd
import ccd
import timeit

def run_rbow():
    specs_url = 'http://localhost'
    chips_url = 'http://localhost'
    requested_ubids = ('LANDSAT_4/TM/SRB1','LANDSAT_4/TM/SRB2', 'LANDSAT_4/TM/SRB3', 'LANDSAT_4/TM/SRB4',
                       'LANDSAT_4/TM/SRB5', 'LANDSAT_4/TM/BTB6', 'LANDSAT_4/TM/SRB7', 'LANDSAT_4/TM/PIXELQA',
                       'LANDSAT_5/TM/SRB1', 'LANDSAT_5/TM/SRB2', 'LANDSAT_5/TM/SRB3', 'LANDSAT_5/TM/SRB4',
                       'LANDSAT_5/TM/SRB5', 'LANDSAT_5/TM/BTB6', 'LANDSAT_5/TM/SRB7', 'LANDSAT_5/TM/PIXELQA',
                       'LANDSAT_7/ETM/SRB1', 'LANDSAT_7/ETM/SRB2', 'LANDSAT_7/ETM/SRB3', 'LANDSAT_7/ETM/SRB4',
                       'LANDSAT_7/ETM/SRB5', 'LANDSAT_7/ETM/BTB6', 'LANDSAT_7/ETM/SRB7', 'LANDSAT_7/ETM/PIXELQA',
                       'LANDSAT_8/OLI_TIRS/SRB2', 'LANDSAT_8/OLI_TIRS/SRB3', 'LANDSAT_8/OLI_TIRS/SRB4',
                       'LANDSAT_8/OLI_TIRS/SRB5', 'LANDSAT_8/OLI_TIRS/SRB6', 'LANDSAT_8/OLI_TIRS/SRB7',
                       'LANDSAT_8/OLI_TIRS/BTB10', 'LANDSAT_8/OLI_TIRS/PIXELQA')
    return rainbow(-2094585, 1682805, '1982-01-01/2015-12-31', specs_url, chips_url, requested_ubids)


def run_detect(rbow):
    def dtstr_to_ordinal(dtstr, iso=True):
        """ Return ordinal from string formatted date"""
        _fmt = '%Y-%m-%dT%H:%M:%SZ' if iso else '%Y-%m-%d %H:%M:%S'
        _dt = datetime.strptime(dtstr, _fmt)
        return _dt.toordinal()
        #return datetime.strptime(dtstr, _fmt).toordinal()

    row, col = 97, 57
    rainbow_date_array = np.array(rbow['t'].values)
    result = ccd.detect([dtstr_to_ordinal(str(pd.to_datetime(i)), False) for i in rainbow_date_array],
                        np.array(rbow['blue'].values[:, row, col]),
                        np.array(rbow['green'].values[:, row, col]),
                        np.array(rbow['red'].values[:, row, col]),
                        np.array(rbow['nir'].values[:, row, col]),
                        np.array(rbow['swir1'].values[:, row, col]),
                        np.array(rbow['swir2'].values[:, row, col]),
                        np.array(rbow['thermal'].values[:, row, col]),
                        np.array(rbow['cfmask'].values[:, row, col], dtype=int),
                        params={})
    return True


if __name__ == '__main__':
    iters = 5
    t = timeit.Timer("run_detect(r)", setup="from __main__ import run_detect, run_rbow; r=run_rbow()")
    print(t.timeit(iters)/iters)
    #run_rbow()



