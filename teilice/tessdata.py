import os
import re
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

from . import common as cm

#from .tesslightcurve import TessLightCurve

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

class TessLightCurve(object):
    def __int__(self):
        pass

def read_lc(filename, fluxkey='PDCSAP_FLUX'):
    hdulst = fits.open(filename)
    data1 = hdulst[1].data
    data2 = hdulst[2].data
    hdulst.close()

    t_lst = data1['TIME']
    q_lst = data1['QUALITY']
    flux_lst = data1[fluxkey]
    cenx_lst = data1['MOM_CENTR1']
    ceny_lst = data1['MOM_CENTR2']
    tcorr_lst = data1['TIMECORR']

    m1 = q_lst==0
    # filter NaN values
    m2 = ~np.isnan(t_lst)
    m3 = ~np.isnan(flux_lst)
    m = m1*m2*m3

    #t_lst = t_lst[m]
    #f_lst = f_lst[m]
    #cenx_lst = cenx_lst[m]
    #ceny_lst = ceny_lst[m]

    shape = data2.shape
    aperture = data2&2>0
    bkgmask  = data2&4>0

    #return (t_lst, q_lst, flux_lst, cenx_lst, ceny_lst, tcorr_lst,
    #        shape, aperture, bkgmask)

    tesslc = TessLightCurve()
    tesslc.t_lst         = t_lst
    tesslc.q_lst         = q_lst
    tesslc.flux_lst      = flux_lst
    tesslc.cenx_lst      = cenx_lst
    tesslc.ceny_lst      = ceny_lst
    tesslc.tcorr_lst     = data1['TIMECORR']
    tesslc.pos_corr1_lst = data1['POS_CORR1']
    tesslc.pos_corr2_lst = data1['POS_CORR2']
    tesslc.shape = data2.shape
    tesslc.aperture = aperture
    tesslc.bkgmask = bkgmask

    return tesslc

def read_tp(filename):
    hdulst = fits.open(filename)
    data1 = hdulst[1].data
    head1 = hdulst[1].header
    data2 = hdulst[2].data
    hdulst.close()

    #mask1 = data1['QUALITY']==0
    #mask2 = ~np.isnan(data1['TIME'])
    #data1 = data1[mask1*mask2]
    t_lst     = data1['TIME']
    q_lst     = data1['QUALITY']
    image_lst = data1['FLUX']
    aperture = data2&2>0

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

    return t_lst, q_lst, image_lst, aperture, wcoord

def make_twomin_target_lst():

    pattern = '[a-zA-Z\-\s]+tess(\d{13})\-s(\d{4})\-(\d{16})\-\d{4}\-s_lc\.fits'

    path = os.path.join(cm.DOWNSCRIPT_PATH, 'sector')
    tic_lst = {}
    for fname in sorted(os.listdir(path)):
        mobj = re.match('tesscurl_sector_(\d+)_lc\.sh', fname)
        if mobj:
            sector = int(mobj.group(1))
            filename = os.path.join(path, fname)
            infile = open(filename)
            for row in infile:
                mobj2 = re.match(pattern, row)
                if mobj2:
                    tic = int(mobj2.group(3))
                    if tic not in tic_lst:
                        tic_lst[tic] = []
                    tic_lst[tic].append(sector)
            infile.close()

    # write the index to file
    outfilename = os.path.join(cm.CACHE_PATH, 'tess_targets_2min.dat')
    outfile = open(outfilename, 'w')
    for tic, sector_lst in sorted(tic_lst.items()):
        outfile.write('{:11d}:'.format(tic))
        string_lst = [str(s) for s in sorted(sector_lst)]
        outfile.write(','.join(string_lst)+os.linesep)
    outfile.close()
