import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.ticker as tck

import astropy.units as u
from astropy.wcs import WCS
from astroquery.skyview import SkyView


def find_best_bc(bcx_lst, bcy_lst):
    bcx_med = np.median(bcx_lst)
    bcy_med = np.median(bcy_lst)
    dist_lst = (bcx_lst-bcx_med)**2 + (bcy_lst-bcy_med)**2
    return dist_lst.argmin()

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


class Tesscut_LC(Figure):
    def __init__(self, tesslc, *args, **kwargs):
        Figure.__init__(self, *args, **kwargs)
        self.canvas = FigureCanvasAgg(self)

        tesscutimg = tesslc.tesscutimg
        ax1 = self.add_axes([0.06, 0.10, 0.40, 0.80],
                    projection=tesscutimg.wcoord)
        ax2 = self.add_axes([0.58, 0.66, 0.38, 0.23])
        ax3 = self.add_axes([0.58, 0.38, 0.38, 0.23])
        ax4 = self.add_axes([0.58, 0.10, 0.38, 0.23])
        axc = self.add_axes([0.45, 0.10, 0.01, 0.80])

        m = tesslc.q_lst==0
        i = find_best_bc(tesslc.bcx_lst[m], tesslc.bcy_lst[m])

        cax = ax1.imshow(tesscutimg.fluxarray[m][i],
                vmin=tesscutimg.vmin, vmax=tesscutimg.vmax, cmap='YlGnBu_r')
        _x1, _x2 = ax1.get_xlim()
        _y1, _y2 = ax1.get_ylim()

        # plot aperture
        bound_lst = get_aperture_bound(tesscutimg.aperture)
        for (x1, y1, x2, y2) in bound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-')

        # plot background mask
        bkgbound_lst = get_aperture_bound(tesscutimg.bkgmask)
        for (x1, y1, x2, y2) in bkgbound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r--', lw=0.5)

        ax1.grid(True, color='w', ls='--', lw=0.5)

        # plot nearby stars
        tictable = tesslc.target.tictable
        mask = tictable['Tmag']<16
        newtictable = tictable[mask]
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

        self.colorbar(cax, cax=axc)

        # plot light curve
        ax2.plot(tesslc.t_lst[m], tesslc.flux_lst[m], 'o', c='C0',
                ms=1, mew=0, alpha=0.6, label='Flux')
        ax2.plot(tesslc.t_lst[m], tesslc.bkg_lst[m], 'o', c='C1',
                ms=1, mew=0, alpha=0.6, label='Background')
        ax3.plot(tesslc.t_lst[m], tesslc.fluxcorr_lst[m], 'o', c='C2',
                ms=1, mew=0, alpha=0.6, label='Corrected Flux')

        # plot barycenter movement
        ax4.plot(tesslc.t_lst[m], tesslc.bcx_lst[m], 'o', c='C0',
                ms=0.5, mew=0, alpha=0.6)
        ax4.plot(tesslc.t_lst[m], tesslc.bcy_lst[m], 'o', c='C1',
                ms=0.5, mew=0, alpha=0.6)
        #ax2.legend(loc='upper left')
        #ax3.legend()

        # adjust ticks
        for ax in [ax2, ax3, ax4]:
            ax.set_xlim(tesslc.t_lst[m][0], tesslc.t_lst[m][-1])
            ax.xaxis.set_major_locator(tck.MultipleLocator(5))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(1))

        ax2.set_ylabel('Flux and Background')
        ax3.set_ylabel('Corrected Flux')
        ax4.set_ylabel('Barycenter')
        ax4.set_xlabel('Time (BJD-2457000)')

        title = ('TIC {0.tic}'
                 ' (RA={0.ra:9.5f}, Dec={0.dec:+9.5f}, Tmag={0.tmag:.2f})'
                 ' Sector {1.sector}').format(tesslc.target, tesslc)
        self.suptitle(title)

    def close(self):
        plt.close(self)

class Tesscut_Skyview(Figure):

    def __init__(self, tesslc, *args, **kwargs):
        Figure.__init__(self, *args, **kwargs)
        self.canvas = FigureCanvasAgg(self)

        tesscutimg = tesslc.tesscutimg
        ax1 = self.add_axes([0.06, 0.10, 0.40, 0.80],
                    projection=tesscutimg.wcoord)

        m = tesslc.q_lst==0
        i = find_best_bc(tesslc.bcx_lst[m], tesslc.bcy_lst[m])

        cax = ax1.imshow(tesscutimg.fluxarray[m][i],
                vmin=tesscutimg.vmin, vmax=tesscutimg.vmax, cmap='YlGnBu_r')
        _x1, _x2 = ax1.get_xlim()
        _y1, _y2 = ax1.get_ylim()

        # plot aperture
        bound_lst = get_aperture_bound(tesscutimg.aperture)
        for (x1, y1, x2, y2) in bound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-')

        # plot background mask
        bkgbound_lst = get_aperture_bound(tesscutimg.bkgmask)
        for (x1, y1, x2, y2) in bkgbound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r--', lw=0.5)

        ax1.grid(True, color='w', ls='--', lw=0.5)

        # plot nearby stars
        tictable = tesslc.target.tictable
        mask = tictable['Tmag']<16
        newtictable = tictable[mask]
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

        # get sky image of nearby region
        radius = max(tesslc.xsize, tesslc.ysize)*2*21  # in unit of arcsec
        paths = SkyView.get_images(position=tesslc.target.coord, survey='DSS',
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
        ax2 = self.add_axes([0.55, 0.10, 0.40, 0.80],
                            projection=wcoord2)
        ax2.imshow(data, cmap='gray_r')

        # plot pixel grid
        for iy in np.arange(-0.5, tesslc.ysize, 1):
            x_lst = [-0.5, tesslc.xsize-0.5]
            y_lst = [iy, iy]
            ra_lst, dec_lst = tesscutimg.wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst, 'b-', lw=0.3)
        for ix in np.arange(-0.5, tesslc.xsize, 1):
            x_lst = [ix, ix]
            y_lst = [-0.5, tesslc.ysize-0.5]
            ra_lst, dec_lst = tesscutimg.wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst, 'b-', lw=0.3)

        # plot aperture
        bound_lst = get_aperture_bound(tesscutimg.aperture)
        for (x1, y1, x2, y2) in bound_lst:
            ra_lst, dec_lst = tesscutimg.wcoord.all_pix2world(
                    [x1-0.5,x2-0.5], [y1-0.5,y2-0.5], 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst,  'r-', lw=0.7)


        # plot x and y arrows
        for x_lst, y_lst in [([-1.0, +1.5], [-1.0, -1.0]),
                             ([-1.0, -1.0], [-1.0, +1.5])]:
            ra_lst, dec_lst = tesscutimg.wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            x, dx = x2_lst[0], x2_lst[1]-x2_lst[0]
            y, dy = y2_lst[0], y2_lst[1]-y2_lst[0]
            ax2.arrow(x,y,dx,dy,width=1,color='k', lw=0)

        xcoords = ax2.coords[0]
        ycoords = ax2.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.set_axislabel('RA (deg)')
        ycoords.set_axislabel('Dec (deg)')

        title = ('TIC {0.tic}'
                 ' (RA={0.ra:9.5f}, Dec={0.dec:+9.5f}, Tmag={0.tmag:.2f})'
                 ' Sector {1.sector}').format(tesslc.target, tesslc)
        self.suptitle(title)

    def close(self):
        plt.close(self)

class LC_PDM(Figure):
    def __init__(self, tesslc, *args, **kwargs):
        Figure.__init__(self, *args, **kwargs)
        self.canvas = FigureCanvasAgg(self)
        ax1 = self.add_axes([0.08, 0.52, 0.84, 0.40])
        ax2 = self.add_axes([0.08, 0.10, 0.30, 0.32])

        m = tesslc.q_lst==0
        ax1.plot(tesslc.t_lst[m], tesslc.fluxcorr_lst[m], 'o', c='C0',
                ms=2, mew=0, alpha=0.8)
        ax1.set_xlabel('Time (BJD-2457000)')
        ax1.set_ylabel('Flux')
        ax1.grid(True, ls='--')
        ax1.set_axisbelow(True)
        ax1.set_xlim(tesslc.t_lst[m][0], tesslc.t_lst[m][-1])
        ax1.xaxis.set_major_locator(tck.MultipleLocator(5))
        ax1.xaxis.set_minor_locator(tck.MultipleLocator(1))

        meanf = tesslc.fluxcorr_lst[m].mean()
        y1, y2 = ax1.get_ylim()
        yy1 = y1/meanf
        yy2 = y2/meanf
        ax1c = ax1.twinx()
        ax1c.set_ylim(yy1, yy2)

        # calculte GLS periodogram
        tesslc.get_pdm()
        period_lst = np.logspace(-3, 1, 1000)
        power_lst, winpower_lst = tesslc.pdm.get_power(period=period_lst)
        freq_lst = 1/period_lst
        # plot periodogram
        ax2.plot(freq_lst, power_lst, '-', c='C0', lw=0.8, alpha=1)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        _x1, _x2 = 1e-1, 1e3
        _y1, _y2 = ax2.get_ylim()
        # plot cadence
        ax2.axvline(24*60/tesslc.cadence, c='k', ls='--', lw=0.5)
        ax2.text(10**(0.95*math.log10(_x1)+0.05*math.log10(_x2)),
                10**(0.1*math.log10(_y1)+0.9*math.log10(_y2)),
                'cadence={}min'.format(tesslc.cadence))
        # adjust axes
        #ax2.set_xlim(freq_lst[0], freq_lst[-1])
        ax2.set_xlim(_x1, _x2)
        ax2.set_xlabel('Freq (c/d)')
        ax2.set_ylabel('Power')
        ax2.grid(True, ls='--')
        ax2.set_axisbelow(True)

        title = ('TIC {0.tic}'
                 ' (RA={0.ra:9.5f}, Dec={0.dec:+9.5f}, Tmag={0.tmag:.2f})'
                 ' Sector {1.sector}').format(tesslc.target, tesslc)
        self.suptitle(title)

    def close(self):
        plt.close(self)


def make_movie(tesslc, tesscutimg, videoname):

    tesslc.get_nearbystars(r=250)
    mask = tesslc.tictable['Tmag']<16
    newtictable = tesslc.tictable[mask]
    tmag_lst = newtictable['Tmag']
    ra_lst  = newtictable['RAJ2000']
    dec_lst = newtictable['DEJ2000']
    starx_lst, stary_lst = tesscutimg.wcoord.all_world2pix(ra_lst, dec_lst, 0)

    # photometric aperture and background aperture
    bound_lst = get_aperture_bound(tesscutimg.aperture)
    bkgbound_lst = get_aperture_bound(tesscutimg.bkgmask)

    m = tesslc.q_lst==0

    fig = plt.figure(figsize=(12, 5))

    def update(i):
        fig.clf()
        ax1 = fig.add_axes([0.08, 0.10, 0.40, 0.80],
                projection=tesscutimg.wcoord)
        ax2 = fig.add_axes([0.58, 0.55, 0.37, 0.35])
        ax3 = fig.add_axes([0.58, 0.10, 0.37, 0.35])
        axc = fig.add_axes([0.47, 0.10, 0.01, 0.80])

        cax = ax1.imshow(tesscutimg.fluxarray[m][i],
                vmin=tesscutimg.vmin, vmax=tesscutimg.vmax, cmap='YlGnBu_r')
        _x1, _x2 = ax1.get_xlim()
        _y1, _y2 = ax1.get_ylim()

        # plot aperture
        for (x1, y1, x2, y2) in bound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-')

        # plot background mask
        for (x1, y1, x2, y2) in bkgbound_lst:
            ax1.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r--', lw=0.5)

        ax1.grid(True, color='w', ls='--', lw=0.5)

        # plot nearby stars
        ax1.scatter(starx_lst, stary_lst, s=(16-tmag_lst)*20,
                    c='none', ec='r', lw=1)

        # adjust ax1
        ax1.text(0.9*_x1+0.1*_x2, 0.1*_y1+0.9*_y2,
                't={:9.4f}'.format(tesslc.t_lst[m][i]),
                color='w')
        ax1.set_xlim(_x1, _x2)
        ax1.set_ylim(_y1, _y2)
        xcoords = ax1.coords[0]
        ycoords = ax1.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.set_axislabel('RA (deg)')
        ycoords.set_axislabel('Dec (deg)')
   
        fig.colorbar(cax, cax=axc)

        # plot light curve
        ax2.plot(tesslc.t_lst[m], tesslc.flux_lst[m], '-', c='C0',
                lw=0.6, alpha=1)
        ax2.plot(tesslc.t_lst[m], tesslc.bkg_lst[m], '-',  c='C1',
                lw=0.6, alpha=1)
        ax2.plot(tesslc.t_lst[m][i], tesslc.flux_lst[m][i], 'o', c='C0',
                ms=4)
        ax2.plot(tesslc.t_lst[m][i], tesslc.bkg_lst[m][i],  'o', c='C1',
                ms=4)
    
        ax3.plot(tesslc.t_lst[m], tesslc.fluxcorr_lst[m], '-', c='C2',
                lw=0.6, alpha=1)
        ax3.plot(tesslc.t_lst[m][i], tesslc.fluxcorr_lst[m][i], 'o', c='C2',
                ms=4)

        # adjust pixel range
        ax2.set_xlim(tesslc.t_lst[m][0], tesslc.t_lst[m][-1])
        ax3.set_xlim(tesslc.t_lst[m][0], tesslc.t_lst[m][-1])

        fig.suptitle('TIC {} (RA={:9.5f}, Dec={:+9.5f}, Tmag={:.2f}) Sector {}'.format(
                tesslc.tic, tesslc.ra, tesslc.dec, tesslc.tmag,
                tesscutimg.sector))
        #print('{} of {}'.format(i, tesscutimg.nsize))

        ratio = min(i/m.sum(), 1.0)
        term_size = os.get_terminal_size()
        nchar = term_size.columns - 60

        string = '>'*int(ratio*nchar)
        string = string.ljust(nchar, '-')
        prompt = 'Making Video'
        string = '\r {:<30s} |{}| ({:6.2f}%)'.format(prompt, string, ratio*100)
        sys.stdout.write(string)
        sys.stdout.flush()

        return cax,

    anim = animation.FuncAnimation(fig, update,
            frames=np.arange(m.sum()), interval=1, blit=False)

    anim.save(videoname, fps=25, extra_args=['-vcodec', 'libx264'])
    print(' \033[92m Completed\033[0m')
