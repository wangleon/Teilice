import os
import re
import math

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.mast import Tesscut

from .tesscutimage import TesscutImage
from .aperture import Aperture
from .visual import Tesscut_LC, Tesscut_Skyview, LC_PDM, make_movie
from .periodogram import GLS

class TessLightCurve(object):

    def __init__(self, target, sector=None, xsize=15, ysize=15, aperture=None):

        self.target = target
        self.xsize = xsize
        self.ysize = ysize
        self.sector = sector

        if aperture is not None:
            self.aperture = Aperture(aperture)

    def apply_aperture(self, tesscutimg):
        ra_lst  = [self.target.ra]
        dec_lst = [self.target.dec]
        x, y = tesscutimg.wcoord.all_world2pix(ra_lst, dec_lst, 0)
        x0 = round(x[0])
        y0 = round(y[0])

        # set the aperture center
        self.aperture_center = (y0, x0)

        self.aperture_mask = np.zeros((tesscutimg.ny, tesscutimg.nx))
        cy = self.aperture.center[0]
        cx = self.aperture.center[1]

        # copy the aperture data to the new aperture mask array
        for iy in np.arange(self.aperture.shape[0]):
            for ix in np.arange(self.aperture.shape[1]):
                self.aperture_mask[y0+iy-cy, x0+ix-cx] = self.aperture.data[iy, ix]

    def get_lc(self, tesscut_path=None):
        """Get light curve.
        """

        # get tesscutfile
        filename, camera, ccd = self.target.get_tesscutfile(
                                sector = self.sector,
                                xsize  = self.xsize,
                                ysize  = self.ysize,
                                tesscut_path = tesscut_path,
                                )
        if filename is None:
            # tesscut file is not in cache and need to be downloaded from MAST
            # archive
            self.target.download_tesscut(self.sector, self.xsize, self.ysize,
                                tesscut_path=tesscut_path)
            filename, camera, ccd = self.target.get_tesscutfile(
                                    sector = self.sector,
                                    xsize  = self.xsize,
                                    ysize  = self.ysize,
                                    tesscut_path = tesscut_path,
                                    )

        self.camera = camera
        self.ccd    = ccd
        self.tesscutimg = TesscutImage(filename)
        self.apply_aperture(self.tesscutimg)

        # set photometric aperture
        aperture = self.aperture_mask
        objmask = aperture>0
        nobj = objmask.sum()
        self.tesscutimg.set_aperture(aperture)

        # determine background aperture
        # find the first layer that QUALITY flag==0
        i = np.nonzero(self.tesscutimg.q_lst==0)[0][0]
        fluximg = self.tesscutimg.fluxarray[i]
        bkgmask = fluximg < np.percentile(fluximg, 50)
        bkgmask = bkgmask*(~objmask)
        nbkg = bkgmask.sum()
        self.tesscutimg.set_bkgmask(bkgmask)

        t_lst, q_lst, fluxsum_lst, bkg_lst = [], [], [], []
        # initialize list of barycenter positions (x, y)
        cenx_lst, ceny_lst = [], []
        for row in self.tesscutimg.table:
            t_lst.append(row['TIME'])
            q_lst.append(row['QUALITY'])
            fluximg = row['FLUX']
            xsum = (fluximg*aperture).sum(axis=0)
            ysum = (fluximg*aperture).sum(axis=1)
            # calculate barycenter
            cenx = (xsum*np.arange(self.tesscutimg.nx)).sum()/(xsum.sum())
            ceny = (ysum*np.arange(self.tesscutimg.ny)).sum()/(ysum.sum())
            fluxsum = (aperture*fluximg).sum()
            mean_bkg = (bkgmask*fluximg).sum()/nbkg
            bkg = mean_bkg*nobj
            
            fluxsum_lst.append(fluxsum)
            bkg_lst.append(bkg)
            cenx_lst.append(cenx)
            ceny_lst.append(ceny)

        self.t_lst = np.array(t_lst)
        self.q_lst = np.array(q_lst)
        self.fluxsum_lst = np.array(fluxsum_lst)
        self.bkg_lst  = np.array(bkg_lst)
        self.cenx_lst  = np.array(cenx_lst)# - self.aperture_center[1]
        self.ceny_lst  = np.array(ceny_lst)# - self.aperture_center[0]
        self.flux_lst = self.fluxsum_lst - self.bkg_lst
        self.tcorr_lst     = self.tesscutimg.table['TIMECORR']
        self.pos_corr1_lst = self.tesscutimg.table['POS_CORR1']
        self.pos_corr2_lst = self.tesscutimg.table['POS_CORR2']

        # find cadence in minutes
        dt = np.median(np.diff(self.t_lst))*24*60
        if abs(dt-2)<0.1:
            self.cadence = 2
        elif abs(dt-10)<0.1:
            self.cadence = 10
        elif abs(dt-30)<0.1:
            self.cadence = 30
        else:
            self.cadence = 0

        #videoname = 'tesscutmovie_{:011d}_s{:04d}_{}_{}.mp4'.format(
        #        self.tic, sector, camera, ccd)
        #make_movie(self, tesscutimg, videoname)

    def get_pdm(self):
        """Get periodogram.
        """
        m = self.q_lst==0
        self.pdm = GLS(self.t_lst[m], self.flux_lst[m])

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

        aperture_mask = self.tesscutimg.aperture_mask & 1
        aperture_mask += np.int32(self.tesscutimg.aperture)*2
        aperture_mask += np.int32(self.tesscutimg.bkgmask)*4

        hdulst = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.BinTableHDU(data=lc_table),
                    fits.ImageHDU(data=aperture_mask),
                    ])
        hdulst.writeto(filename, overwrite=True)

    def plot_tesscut_lc(self, figname=None):
        if figname is None:
            figname = 'tesscut_{:011d}_s{:04d}.png'.format(
                        self.target.tic, self.sector)

        fig = Tesscut_LC(self, figsize=(12, 5), dpi=200)
        fig.savefig(figname)
        fig.close()

    def plot_tesscut_skyview(self, figname=None):
        if figname is None:
            figname = 'tesscut_skyview_{:011d}_s{:04d}.png'.format(
                        self.target.tic, self.sector)

        fig = Tesscut_Skyview(self, figsize=(12, 5), dpi=200)
        fig.savefig(figname)
        fig.close()

    def plot_lc_pdm(self, figname=None):
        if figname is None:
            figname = 'tesslc_pdm_{:011d}_s{:04d}.png'.format(
                        self.target.tic, self.sector)

        fig = LC_PDM(self, figsize=(12, 6), dpi=200)
        fig.savefig(figname)
        fig.close()

    def plot_image2(self, tesscutimg, figname):

        fig = plt.figure(figsize=(12, 5), dpi=200)
        ax1 = fig.add_axes([0.08, 0.10, 0.4, 0.8],
                    projection=tesscutimg.wcoord)

        bcx_med = np.median(self.bcx_lst)
        bcy_med = np.median(self.bcy_lst)
        dist_lst = (self.bcx_lst-bcx_med)**2 + (self.bcy_lst-bcy_med)**2
        i = dist_lst.argmin()
        cax = ax1.imshow(tesscutimg.fluxarray[i],
                vmin=tesscutimg.vmin, vmax=tesscutimg.vmax, cmap='YlGnBu_r')

        # plot aperture
        bound_lst = get_aperture_bound(tesscutimg.aperture)
        for (x1, y1, x2, y2) in bound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-', )

        # plot background mask
        bkgbound_lst = get_aperture_bound(tesscutimg.bkgmask)
        for (x1, y1, x2, y2) in bkgbound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r--', lw=0.5)

        ax1.grid(True, color='w', ls='--', lw=0.5)

        xcoords = ax1.coords[0]
        ycoords = ax1.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.set_axislabel('RA (deg)')
        ycoords.set_axislabel('Dec (deg)')

        # get sky image of nearby region
        radius = max(self.xsize, self.ysize)*2*21  # in unit of arcsec
        paths = SkyView.get_images(position=self.coord, survey='DSS',
                radius  = radius*u.arcsec,
                sampler ='Clip',
                scaling = 'Log',
                pixels  = (400, 400),
                )
        hdu = paths[0][0]
        data = hdu.data
        head = hdu.header
        wcoord2 = WCS(head)

        # add another axes
        ax2 = fig.add_axes([0.52, 0.10, 0.4, 0.8], projection=wcoord2)
        ax2.imshow(data, cmap='gray_r')

        # plot pixel grid
        for iy in np.arange(-0.5, self.ysize, 1):
            x_lst = [-0.5, self.xsize-0.5]
            y_lst = [iy, iy]
            ra_lst, dec_lst = tesscutimg.wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst, 'b-', lw=0.3)
        for ix in np.arange(-0.5, self.xsize, 1):
            x_lst = [ix, ix]
            y_lst = [-0.5, self.ysize-0.5]
            ra_lst, dec_lst = tesscutimg.wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst, 'b-', lw=0.3)

        # plot x and y arrows
        for x_lst, y_lst in [([-1.0, +1.5], [-1.0, -1.0]),
                             ([-1.0, -1.0], [-1.0, +1.5])]:
            ra_lst, dec_lst = tesscutimg.wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            x, dx = x2_lst[0], x2_lst[1]-x2_lst[0]
            y, dy = y2_lst[0], y2_lst[1]-y2_lst[0]
            ax2.arrow(x,y,dx,dy,width=1,color='k', lw=0)


        #x_lst = [0, 0, self.xsize, self.xsize, 0]
        #y_lst = [0, self.ysize, self.ysize, 0, 0]
        xcoords = ax2.coords[0]
        ycoords = ax2.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.set_axislabel('RA (deg)')
        ycoords.set_axislabel('Dec (deg)')

        fig.savefig(figname)
        plt.close(fig)

