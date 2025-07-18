import numpy as np
import astropy.io.fits as fits
from astropy.table import Table

from .periodogram import GLS
from . import utils
from .tessdata import read_lc
from .visual import LC_PDM

class TessLightCurve(object):
    def __int__(self):
        pass

    def get_tictable(self):
        pass

    @staticmethod
    def from_lc_file(lc_filename):
        result = read_lc(lc_filename)

        tesslc = TessLightCurve()
        tesslc.t_lst        = result[0]
        tesslc.q_lst        = result[1]
        tesslc.flux_lst     = result[2]
        tesslc.cenx_lst     = result[3]
        tesslc.ceny_lst     = result[4]
        tesslc.tcorr_lst    = result[5]
        #tesslc.pos_corr1_lst = data1['POS_CORR1']
        #tesslc.pos_corr2_lst = data1['POS_CORR2']
        tesslc.shape        = result[6]
        tesslc.aperture     = result[7]
        tesslc.bkgmask      = result[8]

        return tesslc


    def save_fits(self, filename):
        """Save the light curve into FITS file.

        Args:
            filename (str): Filename of output FITS.
        """

        lc_table = Table()
        lc_table.add_column(self.t_lst,     name='TIME')
        lc_table.add_column(self.tcorr_lst, name='TIMECORR')
        lc_table.add_column(self.flux_lst,  name='SAP_FLUX')
        lc_table.add_column(self.bkg_lst,   name='SAP_BKG')
        lc_table.add_column(self.q_lst,     name='QUALITY')
        lc_table.add_column(self.cenx_lst,  name='MOM_CENTR1')
        lc_table.add_column(self.ceny_lst,  name='MOM_CENTR2')
        lc_table.add_column(self.pos_corr1_lst, name='POS_CORR1')
        lc_table.add_column(self.pos_corr2_lst, name='POS_CORR2')

        aperture_mask = np.ones(self.shape, dtype=np.int32)
        aperture_mask += np.int32(self.aperture)*2
        aperture_mask += np.int32(self.bkgmask)*4

        hdulst = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.BinTableHDU(data=lc_table),
                    fits.ImageHDU(data=aperture_mask),
                    ])
        hdulst.writeto(filename, overwrite=True)

    def get_pdm(self):
        """Get periodogram.
        """
        m = self.q_lst==0
        self.pdm = GLS(self.t_lst[m], self.flux_lst[m])

    def plot_lc_pdm(self, figname=None):
        if figname is None:
            figname = 'tesslc_pdm_{:011d}_s{:04d}.png'.format(
                        self.target.tic, self.sector)

        fig = LC_PDM(self, figsize=(12, 6), dpi=200)
        fig.savefig(figname)
        fig.close()
