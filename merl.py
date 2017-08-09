from merlin.support import aardvark, chip_spec_queries, data_config
from merlin.timeseries import pyccd as pyccd_rods
import timeit


def get_rods():
    dc       = data_config()
    point    = (dc['x'], dc['y'])
    specurl  = 'http://localhost'
    chipurl  = 'http://localhost'
    specs_fn = aardvark.chip_specs
    chips_fn = aardvark.chips
    acquired = dc['acquired']
    queries  = chip_spec_queries(chipurl)
    pi = pyccd_rods(point, specurl, specs_fn, chipurl, chips_fn, acquired, queries)
    return True


if __name__ == '__main__':
    iters = 5
    t = timeit.Timer("get_rods()", setup="from __main__ import get_rods")
    print(t.timeit(iters)/iters)



