import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

class TesscutImage():
    def __init__(self, filename):
        self.filename = filename

        hdulst = fits.open(filename)
        self.table = hdulst[1].data
        self.header1 = hdulst[1].header
        self.header2 = hdulst[2].header
        hdulst.close()

        self.nsize = len(self.table)
        # get image shape
        self.ny, self.nx = self.table[0]['FLUX'].shape
        self.fluxarray = np.array([self.table[i]['FLUX']
                            for i in range(self.nsize)])
        self.vmin = np.percentile(self.fluxarray, 10)
        self.vmax = np.percentile(self.fluxarray, 95)

        wcs_input_dict = {
            'CTYPE1': self.header1['1CTYP5'],
            'CUNIT1': self.header1['1CUNI5'],
            'CDELT1': self.header1['1CDLT5'],
            'CRPIX1': self.header1['1CRPX5'],
            'CRVAL1': self.header1['1CRVL5'],
            'NAXIS1': self.nx,
            'CTYPE2': self.header1['2CTYP5'],
            'CUNIT2': self.header1['2CUNI5'],
            'CDELT2': self.header1['2CDLT5'],
            'CRPIX2': self.header1['2CRPX5'],
            'CRVAL2': self.header1['2CRVL5'],
            'NAXIS2': self.ny,
            'PC1_1':  self.header1['11PC5'],
            'PC1_2':  self.header1['12PC5'],
            'PC2_1':  self.header1['21PC5'],
            'PC2_2':  self.header1['22PC5'],
            }
        self.wcoord = WCS(wcs_input_dict)

    def plot_axes(self, axes, index):
        pass
    def set_aperture(self, aperture):
        self.aperture = aperture
    def set_bkgmask(self, bkgmask):
        self.bkgmask = bkgmask
