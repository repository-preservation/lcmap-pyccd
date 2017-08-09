from datetime import datetime
from test.shared import snap, rainbow

import timeit

# detect results for x, y: -2094435, 1681005 took 2.455234 seconds to generate  VERIFIED 2.6 seconds locally
# detect results for x, y: -2033595, 1685955 took 1.879693 seconds to generate
# detect results for x, y: -2098875, 1694895 took 0.986887 seconds to generate  VERIFIED 1.8 seconds locally


def run_rbow():
    pixl_x, pixl_y = -2094435, 1681005
    pixl_x, pixl_y = -2098875, 1694895
    inputs_url = "https://lcmaphost.com?x={x}&y={y}&acquired=1982-01-01/2015-12-31\
    &ubid=LANDSAT_4/TM/SRB1&ubid=LANDSAT_4/TM/SRB2&ubid=LANDSAT_4/TM/SRB3&ubid=LANDSAT_4/TM/SRB4\
    &ubid=LANDSAT_4/TM/SRB5&ubid=LANDSAT_4/TM/BTB6&ubid=LANDSAT_4/TM/SRB7&ubid=LANDSAT_4/TM/PIXELQA\
    &ubid=LANDSAT_5/TM/SRB1&ubid=LANDSAT_5/TM/SRB2&ubid=LANDSAT_5/TM/SRB3&ubid=LANDSAT_5/TM/SRB4\
    &ubid=LANDSAT_5/TM/SRB5&ubid=LANDSAT_5/TM/BTB6&ubid=LANDSAT_5/TM/SRB7&ubid=LANDSAT_5/TM/PIXELQA\
    &ubid=LANDSAT_7/ETM/SRB1&ubid=LANDSAT_7/ETM/SRB2&ubid=LANDSAT_7/ETM/SRB3&ubid=LANDSAT_7/ETM/SRB4\
    &ubid=LANDSAT_7/ETM/SRB5&ubid=LANDSAT_7/ETM/BTB6&ubid=LANDSAT_7/ETM/SRB7&ubid=LANDSAT_7/ETM/PIXELQA\
    &ubid=LANDSAT_8/OLI_TIRS/SRB2&ubid=LANDSAT_8/OLI_TIRS/SRB3&ubid=LANDSAT_8/OLI_TIRS/SRB4&ubid=LANDSAT_8/OLI_TIRS/SRB5\
    &ubid=LANDSAT_8/OLI_TIRS/SRB6&ubid=LANDSAT_8/OLI_TIRS/SRB7&ubid=LANDSAT_8/OLI_TIRS/BTB10\
    &ubid=LANDSAT_8/OLI_TIRS/PIXELQA".format(x=pixl_x, y=pixl_y)
    #dates           = [i.split('=')[1] for i in inputs_url.split('&') if 'acquired=' in i][0]
    chips_url       = inputs_url.split('?')[0]
    specs_url       = chips_url.replace('/chips', '/chip-specs')
    querystr_list   = inputs_url.split('?')[1].split('&')
    requested_ubids = tuple([i.replace('ubid=', '') for i in querystr_list if 'ubid=' in i])
    # get chip id
    chip_x, chip_y = snap(pixl_x, pixl_y)
    xindex = int((pixl_x - chip_x) / 30)
    yindex = int((chip_y - pixl_y) / 30)
    _start = datetime.now()
    rbow = rainbow(-2094585, 1682805, '1982-01-01/2015-12-31', specs_url, chips_url, requested_ubids)
    return True

if __name__ == '__main__':
    iters = 5
    t = timeit.Timer("run_rbow()", setup="from __main__ import run_rbow")
    print(t.timeit(iters)/iters)
    #run_rbow()



