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
from astroquery.vizier import Vizier
from astroquery.mast import Tesscut
from astroquery.skyview import SkyView

from .tesscutimage import TesscutImage
from .aperture import Aperture
cache_path = os.path.expanduser('~/.teilice')


def get_aperture_bound(aperture):
    bound_lst = []
    ny, nx = aperture.shape
    for iy in np.arange(ny):
        for ix in np.arange(nx):
            if aperture[iy, ix]>0:
                line_lst = [(ix, iy,   ix, iy+1),
                            (ix, iy+1, ix+1, iy+1),
                            (ix+1, iy, ix+1, iy+1),
                            (ix, iy,   ix+1, iy),
                            ]
                for line in line_lst:
                    if line in bound_lst:
                        index = bound_lst.index(line)
                        bound_lst.pop(index)
                    else:
                        bound_lst.append(line)
    return bound_lst

def get_sourceinfo(tic):
    catid = 'IV/38/tic'
    tablelist = Vizier(catalog=catid, columns=['**'],
                column_filters={'TIC': '={}'.format(tic)}
                ).query_constraints()
    tictable = tablelist[catid]
    row = tictable[0]
    return {'ra':   row['RAJ2000'],
            'dec':  row['DEJ2000'],
            'Tmag': row['Tmag'],
            }

def query_nearbystars(coord, r):
    catid = 'IV/38/tic'
    viz = Vizier(catalog=catid, columns=['**', '+_r'])
    viz.ROW_LIMIT=-1
    tablelist = viz.query_region(coord, radius=r*u.arcsec,
                #column_filters={'Tmag': '<{}'.format(limit_mag)}
                )
    tictable = tablelist[catid]
    return tictable
    #filename = 'tic_nearby_{:011d}.vot'.format(tic)
    #tictable.write(filename, format='votable', overwrite=True)


class TessLightCurve(object):

    nearby_path = os.path.join(cache_path, 'nearby')
    tesscut_path = os.path.join(cache_path, 'tesscut')


    def __init__(self, tic, xsize=15, ysize=15, aperture=None):

        # check existence of cache folders
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        if not os.path.exists(self.nearby_path):
            os.mkdir(self.nearby_path)
        if not os.path.exists(self.tesscut_path):
            os.mkdir(self.tesscut_path)

        # get star info
        self.tic = tic
        info = get_sourceinfo(tic)
        self.ra = info['ra']
        self.dec = info['dec']
        self.tmag = info['Tmag']
        self.coord = SkyCoord(self.ra, self.dec, unit='deg')

        self.xsize = xsize
        self.ysize = ysize

        if aperture is not None:
            self.aperture = Aperture(aperture)

    def get_tesscutfile(self):
        pattern = 'tess\-s(\d{4})\-(\d)\-(\d)_(\d+\.\d+)_(\-?\d+\.\d+)_\d+x\d+_astrocut\.fits'
        filename_lst = []
        for fname in os.listdir(self.tesscut_path):
            mobj = re.match(pattern, fname)
            if mobj:
                sector = int(mobj.group(1))
                camera = int(mobj.group(2))
                ccd    = int(mobj.group(3))
                _ra    = float(mobj.group(4))
                _dec   = float(mobj.group(5))
                _coord = SkyCoord(_ra, _dec, unit='deg')
                _sep = self.coord.separation(_coord)
                _r = _sep.deg*3600
                if _r < 1:
                    filename = os.path.join(self.tesscut_path, fname)
                    row = (filename, sector, camera, ccd)
                    filename_lst.append(row)

        return filename_lst

    def download_tesscut(self):
        #website = 'https://mast.stsci.edu/tesscut'
        #url = '{}/api/v0.1/astrocut?ra={}&dec={}&y={}&x={}'.format(
        #        website, self.ra, self.dec, self.ysize, self.xsize)
        #outfile = 'tesscut_tic{:011d}_{}x{}.zip'.format(
        #        self.tic, self.xsize, self.ysize)
        #outfilename = os.path.join(self.tesscut_path, outfile)
        ##command = 'wget -O {} --content-disposition "{}"'.format(outfilename, url)
        #command = 'wget --content-disposition "{}"'.format(url)
        #os.system(command)
        #command = 'unzip {} -d {}'.format(outfilename, self.tesscut_path)
        #os.system(command)
        Tesscut.download_cutouts(coordinates=self.coord,
                size=(self.ysize, self.xsize), path=self.tesscut_path,
                inflate=True)


    def get_nearbystars(self, r=250):
        r = math.ceil(r)
        found_cache = False

        if not os.path.exists(self.nearby_path):
            os.mkdir(self.nearby_path)

        for fname in os.listdir(self.nearby_path):
            mobj = re.match('tic_nearby_(\d+)_r(\d+)s\.vot', fname)
            if mobj:
                _tic = int(mobj.group(1))
                _r   = int(mobj.group(2))
                if _tic==self.tic and _r==r:
                    found_cache = True
                    break
        if found_cache:
            filename = os.path.join(self.nearby_path, fname)
            tictable = Table.read(filename)
        else:
            tictable = query_nearbystars(self.coord, r)
            fname = 'tic_nearby_{:011d}_r{:d}s.vot'.format(self.tic, r)
            filename = os.path.join(self.nearby_path, fname)
            tictable.write(filename, format='votable', overwrite=True)

        self.tictable = tictable

    def apply_aperture(self, tesscutimg):
        x, y = tesscutimg.wcoord.all_world2pix([self.ra], [self.dec], 0)
        x0 = int(x)+1
        y0 = int(y)+1
        self.aperture_mask = np.zeros((tesscutimg.ny, tesscutimg.nx))
        cy = self.aperture.center[0]
        cx = self.aperture.center[1]

        # copy the aperture data to the new aperture mask array
        for iy in np.arange(self.aperture.shape[0]):
            for ix in np.arange(self.aperture.shape[1]):
                self.aperture_mask[y0+iy-cy, x0+ix-cx] = self.aperture.data[iy, ix]

    def get_lc(self):

        # get tesscutfile
        filename_lst = self.get_tesscutfile()
        if len(filename_lst)==0:
            self.download_tesscut()
            filename_lst = self.get_tesscutfile()
        print(filename_lst)

        for filename, sector, camera, ccd in filename_lst:
            tesscutimg = TesscutImage(filename)
            self.apply_aperture(tesscutimg)

            aperture = self.aperture_mask
            objmask = aperture>0
            nobj = objmask.sum()
            tesscutimg.set_aperture(aperture)

            fluximg = tesscutimg.fluxarray[0]
            bkgmask = fluximg < np.percentile(fluximg, 50)
            bkgmask = bkgmask*(~objmask)
            nbkg = bkgmask.sum()
            tesscutimg.set_bkgmask(bkgmask)

            t_lst, flux_lst, bkg_lst = [], [], []
            bcx_lst, bcy_lst = [], []
            for row in tesscutimg.table:
                if row['QUALITY']>0:
                    continue
                t_lst.append(row['TIME'])
                fluximg = row['FLUX']
                xsum = (fluximg*aperture).sum(axis=0)
                ysum = (fluximg*aperture).sum(axis=1)
                bcx = (xsum*np.arange(tesscutimg.nx)).sum()/(xsum.sum())
                bcy = (ysum*np.arange(tesscutimg.ny)).sum()/(ysum.sum())
                flux = (aperture*fluximg).sum()
                mean_bkg = (bkgmask*fluximg).sum()/nbkg
                bkg = mean_bkg*nobj
                
                flux_lst.append(flux)
                bkg_lst.append(bkg)
                bcx_lst.append(bcx)
                bcy_lst.append(bcy)
    
            self.t_lst = np.array(t_lst)
            self.flux_lst = np.array(flux_lst)
            self.bkg_lst  = np.array(bkg_lst)
            self.bcx_lst  = np.array(bcx_lst)
            self.bcy_lst  = np.array(bcy_lst)
            self.fluxcorr_lst = self.flux_lst - self.bkg_lst

            lc_table = Table(dtype=[
                    ('TIME',        '<f4'),
                    ('FLUX_RAW',    '<f4'),
                    ('BKG',         '<f4'),
                    ('FLUX_CORR',   '<f4'),
                    ('BCX',         '<f4'),
                    ('BCY',         '<f4'),
                    ])
            for t, f, b, fc, cx, cy in zip(
                    self.t_lst,
                    self.flux_lst,
                    self.bkg_lst,
                    self.fluxcorr_lst,
                    self.bcx_lst,
                    self.bcy_lst):
                lc_table.add_row((t, f, b, fc, cx, cy))
            hdulst = fits.HDUList([
                        fits.PrimaryHDU(),
                        fits.BinTableHDU(data=lc_table),
                        ])
            outfilename = 'tesslc_{:011d}_s{:04d}_{}_{}.fits'.format(
                        self.tic, sector, camera, ccd)
            hdulst.writeto(outfilename, overwrite=True)


            #self.plot_lc_image(tesscutimg, 'tesscut_{:011d}.png'.format(self.tic))
            self.plot_image2(tesscutimg, 'tesscut2_{:011d}.png'.format(self.tic))


    def plot_lc_image(self, tesscutimg, figname):

        fig = plt.figure(figsize=(12, 5), dpi=200)
        ax1 = fig.add_axes([0.08, 0.10, 0.4, 0.8],
                    projection=tesscutimg.wcoord)
        ax2  = fig.add_axes([0.60, 0.55, 0.35, 0.35])
        ax3 = fig.add_axes([0.60, 0.10, 0.35, 0.35])
        axc = fig.add_axes([0.47, 0.10, 0.01, 0.8])

        bcx_med = np.median(self.bcx_lst)
        bcy_med = np.median(self.bcy_lst)
        dist_lst = (self.bcx_lst-bcx_med)**2 + (self.bcy_lst-bcy_med)**2
        i = dist_lst.argmin()

        cax = ax1.imshow(tesscutimg.fluxarray[i], origin='lower',
                vmin=tesscutimg.vmin, vmax=tesscutimg.vmax, cmap='YlGnBu_r')
        _x1, _x2 = ax1.get_xlim()
        _y1, _y2 = ax1.get_ylim()
        
        # plot aperture
        bound_lst = get_aperture_bound(tesscutimg.aperture)
        for (x1, y1, x2, y2) in bound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-', )

        # plot background mask
        bkgbound_lst = get_aperture_bound(tesscutimg.bkgmask)
        for (x1, y1, x2, y2) in bkgbound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r--', lw=0.5)
        ax1.grid(True, color='w', ls='--', lw=0.5)

        # plot nearby stars
        self.get_nearbystars(r=250)
        mask = self.tictable['Tmag']<16
        newtictable = self.tictable[mask]
        #tictable['_r', 'TIC', 'RAJ2000', 'DEJ2000', 'Tmag'].pprint_all()
        tmag_lst = newtictable['Tmag']
        ra_lst  = newtictable['RAJ2000']
        dec_lst = newtictable['DEJ2000']
        x_lst, y_lst = tesscutimg.wcoord.all_world2pix(ra_lst, dec_lst, 0)
        ax1.scatter(x_lst, y_lst, s=(16-tmag_lst)*20,
                    c='none', ec='r', lw=1)

        # adjust ax1
        ax1.set_xlim(_x1, _x2)
        ax1.set_ylim(_y1, _y2)
        xcoords = ax1.coords[0]
        ycoords = ax1.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.set_axislabel('RA (deg)')
        ycoords.set_axislabel('Dec (deg)')

        # plot light curve
        ax2.plot(self.t_lst, self.flux_lst, '-', c='C0', lw=0.6, alpha=1,
                label='Flux')
        ax2.plot(self.t_lst, self.bkg_lst, '-', c='C1', lw=0.6, alpha=1,
                label='Background')
        ax3.plot(self.t_lst, self.fluxcorr_lst, '-', lw=0.6, alpha=1,
                label='Corrected Flux')
        ax2.legend(loc='upper left')
        ax3.legend()
        fig.colorbar(cax, cax=axc)
        fig.suptitle('TIC {}'.format(self.tic))
        fig.savefig(figname)
        plt.close(fig)


    def plot_lc(self, figname):
        t_lst = self.t_lst
        flux_lst = self.flux_lst
        bkg_lst  = self.bkg_lst
        bcx_lst = self.bcx_lst
        bcy_lst = self.bcy_lst

        fig = plt.figure(figsize=(12, 7), dpi=200)
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax4 = ax3.twinx()
        ax1.plot(t_lst, flux_lst, '-', lw=0.6, alpha=1)
        ax1.plot(t_lst, bkg_lst,  '-', lw=0.6, alpha=1)
        ax2.plot(t_lst, flux_lst-bkg_lst, '-', lw=0.6, alpha=1)
        ax3.plot(t_lst, bcx_lst, '-', c='C0', lw=0.6, alpha=1)
        ax4.plot(t_lst, bcy_lst, '-', c='C1', lw=0.6, alpha=1)
        fig.savefig(figname)
        plt.close(fig)

    def plot_image(self, figname):
        wcoord = self.wcoord
        t_lst = self.t_lst
        fluxarray = self.fluxarray
        flux_lst = self.flux_lst
        bkg_lst  = self.bkg_lst
        bcx_lst  = self.bcx_lst
        bcy_lst  = self.bcy_lst
        tictable = self.tictable
        vmin = self.vmin
        vmax = self.vmax
        aperture = self.aperture
        bkgmask = self.bkgmask
        imin = self.imin

        i = imin


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

        paths = SkyView.get_images(position=self.coord, survey='DSS',
                radius = 15/2*21*u.arcsec*3,
                sampler='Clip',
                scaling = 'Log',
                pixels = (400, 400),
                )
        hdu = paths[0][0]
        data = hdu.data
        head = hdu.header
        wcoord2 = WCS(head)
        ax2 = fig.add_axes([0.52, 0.10, 0.4, 0.8],
                    projection=wcoord2)
        ax2.imshow(data, cmap='gray_r')

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



    def make_animation(self, videoname):
        n = self.n
        ra = self.ra
        dec = self.dec
        t_lst = self.t_lst
        fluxarray = self.fluxarray
        flux_lst = self.flux_lst
        bkg_lst  = self.bkg_lst
        bcx_lst  = self.bcx_lst
        bcy_lst  = self.bcy_lst
        wcoord = self.wcoord
        tictable = self.tictable
        vmin = self.vmin
        vmax = self.vmax
        aperture = self.aperture
        bkgmask = self.bkgmask

        fluxbkg_lst = flux_lst - bkg_lst
        
        fig = plt.figure(figsize=(12, 5))

        # initialize background mask
        bkg_cmap = mcolors.LinearSegmentedColormap.from_list('TransRed',
                   [(1,0,0,0), (1,0,0,0.1)], N=2)

        def update(i):
            fig.clf()
            ax1 = fig.add_axes([0.1, 0.1, 0.4, 0.8], projection=wcoord)
            ax2 = fig.add_axes([0.6, 0.6, 0.35, 0.35])
            ax3 = fig.add_axes([0.6, 0.1, 0.35, 0.35])
            axc = fig.add_axes([0.5, 0.1, 0.01, 0.8])
            t = t_lst[i]
            #mean_bkg  = mean_bkg_lst[i]
            #fluximg_dbkg = fluximg - mean_bkg
            #ax1.imshow(fluximg_dbkg, vmin=vmin-mean_bkg, vmax=vmax-mean_bkg)
            cax = ax1.imshow(fluxarray[i], origin='lower',
                    vmin=vmin, vmax=vmax, cmap='YlGnBu_r')
            _x1, _x2 = ax1.get_xlim()
            _y1, _y2 = ax1.get_ylim()

            # plot aperture
            bound_lst = get_aperture_bound(aperture)
            for (x1, y1, x2, y2) in bound_lst:
                ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-', )

            # plot background mask
            bkgbound_lst = get_aperture_bound(bkgmask)
            for (x1, y1, x2, y2) in bkgbound_lst:
                ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r--', lw=0.5)
            ax1.grid(True, color='w', ls='--', lw=0.5)

            # plot background mask
            #ax1.imshow(bkgmask, interpolation='none', cmap=bkg_cmap)

            # plot nearby stars
            mask = tictable['Tmag']<16
            newtictable = tictable[mask]
            tmag_lst = newtictable['Tmag']
            ra_lst  = newtictable['RAJ2000']
            dec_lst = newtictable['DEJ2000']
            x_lst, y_lst = wcoord.all_world2pix(ra_lst, dec_lst, 0)
            ax1.scatter(x_lst, y_lst, s=(16-tmag_lst)*20,
                        c='none', ec='r', lw=1)

            # adjust ax1
            ax1.set_xlim(_x1, _x2)
            ax1.set_ylim(_y1, _y2)
            xcoords = ax1.coords[0]
            ycoords = ax1.coords[1]
            xcoords.set_major_formatter('d.ddd')
            ycoords.set_major_formatter('d.ddd')
            xcoords.set_axislabel('RA (deg)')
            ycoords.set_axislabel('Dec (deg)')
   
            # plot light curve
            ax2.plot(t_lst, flux_lst, '-', lw=0.6, alpha=1)
            ax2.plot(t, flux_lst[i], 'o', ms=4)
            ax2.plot(t_lst, bkg_lst, '-', lw=0.6, alpha=1)
            ax2.plot(t, bkg_lst[i], 'o', ms=4)
    
            ax3.plot(t_lst, fluxbkg_lst, '-', lw=0.6, alpha=1)
            ax3.plot(t, fluxbkg_lst[i], 'o', ms=4)
            fig.colorbar(cax, cax=axc)
            fig.suptitle('t={:9.4f}'.format(t))
            #fig.savefig('ani_{:04d}.png'.format(i))
            print('{} of {}'.format(i, n))
            return cax,
        anim = animation.FuncAnimation(fig, update,
                frames=np.arange(n), interval=1, blit=False)
        anim.save(videoname, fps=25, extra_args=['-vcodec', 'libx264'])

