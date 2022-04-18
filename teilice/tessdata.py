import os
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

filekey_lst = {
         1: (2018206045859, 120),  2: (2018234235059, 121),
         3: (2018263035959, 123),  4: (2018292075959, 124),
         5: (2018319095959, 125),  6: (2018349182459, 126),
         7: (2019006130736, 131),  8: (2019032160000, 136),
         9: (2019058134432, 139), 10: (2019085135100, 140),
        11: (2019112060037, 143), 12: (2019140104343, 144),
        13: (2019169103026, 146), 14: (2019198215352, 150),
        15: (2019226182529, 151), 16: (2019253231442, 152),
        17: (2019279210107, 161), 18: (2019306063752, 162),
        19: (2019331140908, 164), 20: (2019357164649, 165),
        21: (2020020091053, 167), 22: (2020049080258, 174),
        23: (2020078014623, 177), 24: (2020106103520, 180),
        25: (2020133194932, 182), 26: (2020160202036, 188),
        27: (2020186164531, 189), 28: (2020212050318, 190),
        29: (2020238165205, 193), 30: (2020266004630, 195),
        31: (2020294194027, 198), 32: (2020324010417, 200),
        33: (2020351194500, 203), 34: (2021014023720, 204),
        35: (2021039152502, 205), 36: (2021065132309, 207),
        37: (2021091135823, 208), 38: (2021118034608, 209),
        39: (2021146024351, 210), 40: (2021175071901, 211),
        41: (2021204101404, 212), 42: (2021232031932, 213),
        43: (2021258175143, 214), 44: (2021284114741, 215),
        45: (2021310001228, 216), 46: (2021336043614, 217),
        47: (2021364111932, 218), 48: (2022027120115, 219),
        }

def get_lc_filename(sector, tic):
    timestamp, scid = filekey_lst[sector]
    return 'tess{:13d}-s{:04d}-{:016d}-{:04d}-s_lc.fits'.format(
            timestamp, sector, tic, scid)

def get_tp_filename(sector, tic):
    timestamp, scid = filekey_lst[sector]
    return 'tess{:13d}-s{:04d}-{:016d}-{:04d}-s_tp.fits'.format(
            timestamp, sector, tic, scid)


def read_2min_lc_backup(tic, sector_lst, lc_path):
    lc_lst = {}
    for sector, camera, ccd in sector_lst:
        path = os.path.join(data_path,
                's%03d/%d_%d'%(sector, camera, ccd))
        fname = get_lc_filename(sector, tic)
        lc_filename = os.path.join(path, fname)
        table = fits.getdata(lc_filename)
        t_lst = table['TIME']
        f_lst = table['PDCSAP_FLUX']
        q_lst = table['QUALITY']
        m = q_lst==0
        t_lst = t_lst[m]
        f_lst = f_lst[m]
        # filter NaN values
        m2 = ~np.isnan(f_lst)
        t_lst = t_lst[m2]
        f_lst = f_lst[m2]
        lc_lst[sector] = (t_lst, f_lst)
    return lc_lst

def read_lc(filename):
    table = fits.getdata(filename)
    t_lst = table['TIME']
    f_lst = table['PDCSAP_FLUX']
    q_lst = table['QUALITY']
    m = q_lst==0
    t_lst = t_lst[m]
    f_lst = f_lst[m]
    # filter NaN values
    m2 = ~np.isnan(f_lst)
    t_lst = t_lst[m2]
    f_lst = f_lst[m2]
    return t_lst, f_lst

def read_tp(filename):
    hdulst = fits.open(filename)
    data1 = hdulst[1].data
    head1 = hdulst[1].header
    data2 = hdulst[2].data
    hdulst.close()

    mask1 = data1['QUALITY']==0
    mask2 = ~np.isnan(data1['TIME'])
    data1 = data1[mask1*mask2]
    objmask = data2&2>0

    t_lst = data1['TIME']
    image_lst = data1['FLUX']
    ny, nx = image_lst[0].shape

    wcs_input_dict = {
        'CTYPE1': head1['1CTYP5'],
        'CUNIT1': head1['1CUNI5'],
        'CDELT1': head1['1CDLT5'],
        'CRPIX1': head1['1CRPX5'],
        'CRVAL1': head1['1CRVL5'],
        'NAXIS1': nx,
        'CTYPE2': head1['2CTYP5'],
        'CUNIT2': head1['2CUNI5'],
        'CDELT2': head1['2CDLT5'],
        'CRPIX2': head1['2CRPX5'],
        'CRVAL2': head1['2CRVL5'],
        'NAXIS2': ny,
        'PC1_1':  head1['11PC5'],
        'PC1_2':  head1['12PC5'],
        'PC2_1':  head1['21PC5'],
        'PC2_2':  head1['22PC5'],
        }
    wcoord = WCS(wcs_input_dict)

    return t_lst, image_lst, objmask, wcoord
