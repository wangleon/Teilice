import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.ticker as tck

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astroquery.skyview import SkyView

from . import periodogram

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


class MultiSector_LC(Figure):

    def __init__(self, lc_lst, *args, **kwargs):
        Figure.__init__(self, *args, **kwargs)
        self.canvas = FigureCanvasAgg(self)

        #def multi_lc(tic, lc_lst, ticrow, gaiarow, figname, include_gls=False):

        #fig1 = plt.figure(figsize=(18, 8), dpi=200)
        axlc_lst = []

        segment_lst = []
        sector_lst = sorted(lc_lst.keys())
        segment_lst = np.split(sector_lst, np.where(np.diff(sector_lst) != 1)[0]+1)

        foffset_lst = {}
        for isector, (sector, dataitem) in enumerate(sorted(lc_lst.items())):
            t_lst, f_lst = dataitem
            medf = np.median(f_lst)
            if isector == 0:
                medf0 = medf
            foffset = (medf - medf0)
            foffset_lst[sector] = foffset

        tspan_lst = []
        for segs in segment_lst:
            s1 = segs[0]
            s2 = segs[-1]
            t1 = lc_lst[s1][0][0]
            t2 = lc_lst[s2][0][-1]
            tspan_lst.append(t2-t1)
        tspan_lst = np.array(tspan_lst)

        gap = 0.008
        width_lst = np.array([(0.88 - (len(segment_lst)-1)*gap)/tspan_lst.sum()*tspan
                        for tspan in tspan_lst])

        for iseg, segs in enumerate(segment_lst):
            _left = 0.07 + width_lst[0:iseg].sum() + iseg*gap
            _width = width_lst[iseg]
            ax = self.add_axes([_left, 0.5, _width, 0.43])
            axlc_lst.append(ax)
            for sector in segs:
                color = 'C{}'.format(sector%10)
                t_lst, f_lst = lc_lst[sector]
                foffset = foffset_lst[sector]
                ax.plot(t_lst, f_lst - foffset, 'o', color='C0', ms=1, alpha=0.5)
            s1 = segs[0]
            s2 = segs[-1]
            t1 = lc_lst[s1][0][0]
            t2 = lc_lst[s2][0][-1]
            ax.set_xlim(t1, t2)
    
    
            if len(sector_lst)>=15:
                ax.xaxis.set_major_locator(tck.MultipleLocator(20))
                ax.xaxis.set_minor_locator(tck.MultipleLocator(5))
            elif len(sector_lst)>=5:
                ax.xaxis.set_major_locator(tck.MultipleLocator(10))
                ax.xaxis.set_minor_locator(tck.MultipleLocator(1))
            else:
                ax.xaxis.set_major_locator(tck.MultipleLocator(5))
                ax.xaxis.set_minor_locator(tck.MultipleLocator(1))
    
            ax.grid(True, axis='x', ls='--')
            ax.set_axisbelow(True)
    
            d = 2
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                        linestyle='none', color='k', mec='k', mew=1, clip_on=False)
            if iseg==0:
                ax.set_ylabel('FLUX')
            if iseg>0:
                ax.spines.left.set_visible(False)
                ax.tick_params(labelleft=False)
                ax.set_yticks([])
                ax.plot([0, 0], [0, 1], transform=ax.transAxes, **kwargs)
            if iseg<len(segment_lst)-1:
                ax.spines.right.set_visible(False)
                ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kwargs)
    
        y1 = min([ax.get_ylim()[0] for ax in axlc_lst])
        y2 = max([ax.get_ylim()[1] for ax in axlc_lst])
        for iseg, ax in enumerate(axlc_lst):
            ax.set_ylim(y1, y2)
            segs = segment_lst[iseg]
            s1 = segs[0]
            s2 = segs[-1]
            x1, x2 = ax.get_xlim()
            if len(segs)==1:
                text = 'S{:02d}'.format(s1)
            else:
                text = 'S{:02d}-{:02d}'.format(s1, s2)
            ax.text(0.95*x1+0.05*x2, 0.05*y1+0.95*y2, text)
            if iseg==len(segment_lst)-1:
                ax2 = ax.twinx()
                ax2.spines.left.set_visible(False)
                ax2.set_ylim(y1/medf0, y2/medf0)
    
        if include_gls:
            axgls = self.add_axes([0.7, 0.1, 0.25, 0.34])
            period_lst = np.logspace(-3, 1, 1000)
            freq_lst = 1/period_lst
            for sector, dataitem in sorted(lc_lst.items()):
                t_lst, f_lst = dataitem
                pdm = periodogram.GLS(t_lst, f_lst)
                power, _ = pdm.get_power(period=period_lst)
                color = 'C{}'.format(sector%10)
                axgls.plot(freq_lst, power, '-', lw=0.5, alpha=0.6)
            axgls.set_xscale('log')
            axgls.set_yscale('log')
            axgls.set_xlim(0.1, 1000)
            axgls.set_xlabel('Frequency (c/d)')
            axgls.axvline(1440/2, c='k', ls='--', lw=0.5)
    
        # read catalog record
        hip     = ticrow['HIP']
        tyc     = ticrow['TYC']
        ucac4   = ticrow['UCAC4']
        tmass   = ticrow['_2MASS']
        sdss    = ticrow['objID']
        allwise = ticrow['WISEA']
        gaia2   = ticrow['GAIA']
        apass   = ticrow['APASS']
        kic     = ticrow['KIC']
        vmag = ticrow['Vmag']
        gmag = ticrow['gmag']
        rmag = ticrow['rmag']
        imag = ticrow['imag']
        jmag = ticrow['Jmag']
        hmag = ticrow['Hmag']
        kmag = ticrow['Kmag']
        ra    = ticrow['RAJ2000']
        dec   = ticrow['DEJ2000']
        plx   = ticrow['Plx']
        e_plx = ticrow['e_Plx']
        Gmag  = gaiarow['Gmag']
        bprp0 = gaiarow['BP-RP'] - gaiarow['E_BP-RP_']
        coord = SkyCoord(ra, dec, unit='deg')
        gc = coord.transform_to('galactic')
        l = gc.l.deg
        b = gc.b.deg
        
        teff = ticrow['Teff']
        logg = ticrow['logg']
        fe_h = ticrow['__M_H_']
        rad  = ticrow['Rad']
        mass = ticrow['Mass']
        lumc = ticrow['LClass']
        lum  = ticrow['Lum']
        ebv  = ticrow['E_B-V_']
        
        text1_lst = ['TIC {}'.format(tic)]
        if hip is not np.ma.masked and hip>0:
            text1_lst.append('HIP {}'.format(hip))
        if len(tyc)>0:
            text1_lst.append('TYC {}'.format(tyc))
        if kic is not np.ma.masked and kic>0:
            text1_lst.append('KIC {}'.format(kic))
        if len(ucac4)>0:
            text1_lst.append('UCAC4 {}'.format(ucac4))
        if len(tmass)>0:
            text1_lst.append('2MASS J{}'.format(tmass))
        if len(allwise)>0:
            text1_lst.append('ALLWISE {}'.format(allwise))
        if gaia2 is not np.ma.masked and gaia2>0:
            text1_lst.append('GAIA DR2 {}'.format(gaia2))
    
        text1_lst.append('')
        text1_lst.append('ICRS = {:9.5f}, {:9.5f}'.format(ra, dec))
        text1_lst.append('Gal. = {:9.5f}, {:9.5f}'.format(l, b))
    
        self.text(0.04, 0.38, '\n'.join(text1_lst), fontfamily='monospace',
                ha='left', va='top')
    
        text3_lst = []
    
        _text_lst = []
        if vmag is not np.ma.masked:
            _text_lst.append('V = {:5.2f}'.format(vmag))
        if Gmag is not np.ma.masked:
            _text_lst.append('G = {:5.2f}'.format(Gmag))
        text3_lst.append('  '.join(_text_lst))
    
        _text_lst = []
        if jmag is not np.ma.masked:
            _text_lst.append('J = {:5.2f}'.format(jmag))
        if hmag is not np.ma.masked:
            _text_lst.append('H = {:5.2f}'.format(hmag))
        text3_lst.append('  '.join(_text_lst))
    
        if kmag is not np.ma.masked:
            text3_lst.append('K = {:5.2f}'.format(kmag))
        if vmag is not np.ma.masked and kmag is not np.ma.masked:
            text3_lst.append('V-Ks = {:+5.2f}'.format(vmag-kmag))
        if bprp0 is not np.ma.masked:
            text3_lst.append('G(Bp-Rp) = {:+5.2f}'.format(bprp0))
        text3_lst.append('')
    
        if plx is not np.ma.masked:
            if e_plx is not np.ma.masked:
                text = u'Plx = {:.3f} \xb1 {:.3f} ({:.2f}%)'.format(
                        plx, e_plx, e_plx/plx*100)
            else:
                text = 'Plx = {:8g}'.format(plx)
            text3_lst.append(text)
    
        _text_lst = []
        if teff is not np.ma.masked and teff>0:
            _text_lst.append('Teff = {:5g}'.format(teff))
        if logg is not np.ma.masked and logg>0:
            _text_lst.append('logg = {:3.2f}'.format(logg))
        text3_lst.append('  '.join(_text_lst))

        if fe_h is not np.ma.masked:
            text3_lst.append('[M/H] = {:+4.2f}'.format(fe_h))

        _text_lst = []
        if rad is not np.ma.masked:
            _text_lst.append('R = {:4.2f}'.format(rad))
        if mass is not np.ma.masked:
            _text_lst.append('M = {:4.2f}'.format(mass))
        if lum is not np.ma.masked:
            _text_lst.append('L = {:4.2f}'.format(lum))
        text3_lst.append('  '.join(_text_lst))
    
        if lumc is not np.ma.masked:
            text3_lst.append('Class: {}'.format(lumc))
        if ebv is not np.ma.masked:
            text3_lst.append('E(B-V) = {:5.3f}'.format(ebv))
    
        self.text(0.20, 0.38, '\n'.join(text3_lst), fontfamily='monospace',
                ha='left', va='top')
    
        title = 'TIC {:d}'.format(tic)
        self.suptitle(title)
    
        #fig1.savefig(figname)
        #plt.close(fig1)

def plot_tp_complex(tic, target_lst, tp_path, lc_path, figpath,
        tictable_cache, gaia2table_cache,
        gaiae3table_cache, skyview_cache, overwrite=False):
    catid = 'IV/38/tic'
    tablelist = Vizier(catalog=catid, columns=['**'],
                column_filters={'TIC': '={}'.format(tic)}
                ).query_constraints()
    tictable = tablelist[catid]
    row = tictable[0]
    ra = row['RAJ2000']
    dec = row['DEJ2000']
    tmag = row['Tmag']
    coord = SkyCoord(ra, dec, unit='deg')

    for cache in [tictable_cache, gaia2table_cache, gaiae3table_cache,
                skyview_cache]:
        if not os.path.exists(cache):
            os.mkdir(cache)

    ############## get tic table ##############
    tictablename = os.path.join(tictable_cache,
                    'tic_nearby_{:012d}_250.vot'.format(tic))
    if os.path.exists(tictablename):
        tictable = Table.read(tictablename)
    else:
        catid = 'IV/38/tic'
        viz = Vizier(catalog=catid, columns=['**', '+_r'])
        viz.ROW_LIMIT = -1
        tablelist = viz.query_region(coord, radius=250*u.arcsec)
        tictable = tablelist[catid]
        tictable.write(tictablename, format='votable', overwrite=True)

    ############## get gaia2 table #############
    gaia2tablename = os.path.join(gaia2table_cache,
                    'gaia2_nearby_{:012d}_150.vot'.format(tic))
    if os.path.exists(gaia2tablename):
        gaia2table = Table.read(gaia2tablename)
    else:
        catid = 'I/345/gaia2'
        viz = Vizier(catalog=catid, columns=['**', '+_r'])
        viz.ROW_LIMIT = -1
        tablelist = viz.query_region(coord, radius=150*u.arcsec)
        gaia2table = tablelist[catid]
        gaia2table.write(gaia2tablename, format='votable', overwrite=True)

    ############## get gaiae3 table #############
    gaiae3tablename = os.path.join(gaiae3table_cache,
                    'gaiae3_nearby_{:012d}_150.vot'.format(tic))
    if os.path.exists(gaiae3tablename):
        gaiae3table = Table.read(gaiae3tablename)
    else:
        catid = 'I/350/gaiaedr3'
        viz = Vizier(catalog=catid, columns=['**', '+_r'])
        viz.ROW_LIMIT = -1
        tablelist = viz.query_region(coord, radius=150*u.arcsec)
        gaiae3table = tablelist[catid]
        gaiae3table.write(gaiae3tablename, format='votable', overwrite=True)

    ############## get large skyview image ###########
    dss360_filename = os.path.join(skyview_cache,
                    'skyview_dss_{:012d}_360.fits'.format(tic))
    if os.path.exists(dss360_filename):
        hdulst = fits.open(dss360_filename)
        dss360data = hdulst[0].data
        dss360head = hdulst[0].header
    else:
        # get large DSS image
        radius = 360  # in unit of arcsec
        for i in range(10):
            try:
                paths = SkyView.get_images(position=coord, survey='DSS',
                    radius  = radius*u.arcsec,
                    sampler ='Clip',
                    scaling = 'Log',
                    pixels  = (500, 500),
                    )
            except:
                continue
        hdu = paths[0][0]
        hdulst = paths[0]
        hdulst.writeto(dss360_filename, overwrite=True)
        dss360data = hdu.data
        dss360head = hdu.header

    ############## get small skyview image ###########
    dss100_filename = os.path.join(skyview_cache,
                    'skyview_dss_{:012d}_100.fits'.format(tic))
    if os.path.exists(dss100_filename):
        hdulst = fits.open(dss100_filename)
        dss100data = hdulst[0].data
        dss100head = hdulst[0].header
    else:
        # get large DSS image
        radius = 100  # in unit of arcsec
        for i in range(10):
            try:
                paths = SkyView.get_images(position=coord, survey='DSS',
                    radius  = radius*u.arcsec,
                    sampler ='Clip',
                    scaling = 'Log',
                    pixels  = (500, 500),
                    )
            except:
                continue
        hdu = paths[0][0]
        hdulst = paths[0]
        hdulst.writeto(dss100_filename, overwrite=True)
        dss100data = hdu.data
        dss100head = hdu.header

    ######### get star catalog to be dispalyed ##############
    # find nearby stars
    m1 = (tictable['_r']<10)*(tictable['Tmag']-tmag<3)
    if m1.sum()<=2:
        m1 = (tictable['_r']<30)*(tictable['Tmag']-tmag<5)
    if m1.sum()>4:
        m1[np.nonzero(m1)[0][4:]] = False

    # find nearby bright stars
    m2 = (tictable['_r']<60)*(tictable['Tmag']<tmag+1)
    if m2.sum()<=2:
        m2 = (tictable['_r']<120)*(tictable['Tmag']<tmag+1)
    if m2.sum()<=2:
        m2 = (tictable['_r']<120)*(tictable['Tmag']<tmag+2)
    if m2.sum()>4:
        m2[np.nonzero(m2)[0][4:]] = False
    m = m1 + m2

    text_lst = ['']
    text_lst.append('TIC:')
    fmtstr = '{:>5s} {:>11s} {:7s} {:>5s} {:>5s} {:>5s} {:>5s}'
    text = fmtstr.format(
            'r (")', 'TIC', 'Gaia2', 'Tmag', 'Vmag', 'Kmag', 'V-K')
    text_lst.append(text)

    gaia2_lst = []
    for ticrow in tictable[m][0:8]:
        _r     = ticrow['_r']
        _tic   = ticrow['TIC']
        _gaia2 = ticrow['GAIA']
        _tmag  = ticrow['Tmag']
        _vmag  = ticrow['Vmag']
        _kmag  = ticrow['Kmag']
        _vk    = _vmag - _kmag
        if _gaia2 is not np.ma.masked:
            gaia2_lst.append(_gaia2)

        _r = '{:5.1f}'.format(_r)
        _tic = '{:11d}'.format(_tic)
        if _gaia2 is np.ma.masked:
            _gaia2 = ''
        else:
            _gaia2 = '...'+str(_gaia2)[-4:]
        if _tmag is np.ma.masked:
            _tmag = ''
        else:
            _tmag = '{:5.2f}'.format(_tmag)
        if _vmag is np.ma.masked:
            _vmag = ''
        else:
            _vmag = '{:5.2f}'.format(_vmag)
        if _kmag is np.ma.masked:
            _kmag = ''
        else:
            _kmag = '{:5.2f}'.format(_kmag)
        if _vk is np.ma.masked:
            _vk = ''
        else:
            _vk = '{:5.2f}'.format(_vk)

        text = fmtstr.format(
                _r, _tic, _gaia2,  _tmag, _vmag, _kmag, _vk)
        text_lst.append(text)

    text_lst.append('')
    text_lst.append('Gaia DR2:')
    fmtstr = '{:>5s} {:>19s} {:>5s} {:>5s} {:>5s} {:>5s}'
    text = fmtstr.format(
            'r (")', 'Source', 'Gmag', 'BP-RP', 'Plx', 'e_Plx')
    text_lst.append(text)
    m = [row['Source'] in gaia2_lst for row in gaia2table]
    #m = gaia2table['_r']<10
    for gaiarow in gaia2table[m][0:8]:
        _r = gaiarow['_r']
        _gaia2 = gaiarow['Source']
        _gmag  = gaiarow['Gmag']
        _bprp  = gaiarow['BP-RP']
        _ebprp = gaiarow['E_BP-RP_']
        _bprp0 = _bprp - _ebprp
        _plx   = gaiarow['Plx']
        _eplx  = gaiarow['e_Plx']

        _r = '{:5.1f}'.format(_r)
        _gaia2 = str(_gaia2)
        if _gmag is np.ma.masked:
            _gmag = ''
        else:
            _gmag = '{:5.2f}'.format(_gmag)
        if _bprp0 is np.ma.masked:
            _bprp0 = ''
        else:
            _bprp0 = '{:5.2f}'.format(_bprp0)

        if _plx is np.ma.masked:
            _plx = ''
        else:
            _plx = '{:5.2f}'.format(_plx)
        if _eplx is np.ma.masked:
            _eplx = ''
        else:
            _eplx = '{:5.2f}'.format(_eplx)


        text = fmtstr.format(_r, _gaia2, _gmag, _bprp0,
                _plx, _eplx)
        text_lst.append(text)



    for sector, (cam, ccd) in sorted(target_lst[tic].items()):
        figname = os.path.join(figpath,
                    'tp_{:012d}_s{:03d}.png'.format(tic, sector))
        print(tic, sector, cam, ccd)

        if not overwrite and os.path.exists(figname):
            continue

        ######## read lc file #########
        lcfile = get_lc_file(sector, tic)
        lcfilename = os.path.join(lc_path,
                    's{:03d}/{}_{}'.format(sector, cam, ccd),
                    lcfile)
        data = fits.getdata(lcfilename)
        tlc_lst = data['TIME']
        f_lst = data['PDCSAP_FLUX']
        m1    = data['QUALITY']==0
        m2    = ~np.isnan(f_lst)
        tlc_lst = tlc_lst[m1*m2]
        f_lst   = f_lst[m1*m2]

        ########### read tp file #############
        tpfile = get_tp_file(sector, tic)
        tpfilename = os.path.join(tp_path,
                's{:03d}/{}_{}'.format(sector, cam, ccd),
                    tpfile)
        if not os.path.exists(tpfilename):
            continue

        f = fits.open(tpfilename)
        data1 = f[1].data
        head1 = f[1].header
        data2 = f[2].data
        f.close()

        mask1 = data1['QUALITY']==0
        mask2 = ~np.isnan(data1['TIME'])
        data1 = data1[mask1*mask2]
        objmask = data2&2>0
        t_lst = data1['TIME']

        # determine best frame
        medf = np.median(f_lst)
        idx1 = np.abs(f_lst - medf).argmin()
        t = tlc_lst[idx1]
        idx = np.abs(t_lst - t).argmin()

        image = data1[idx]['FLUX']
        ny, nx = image.shape
        # calculate barycentric
        bcx_lst = []
        bcy_lst = []
        for row in data1:
            xsum = (row['FLUX']*objmask).sum(axis=0)
            ysum = (row['FLUX']*objmask).sum(axis=1)
            # calculate barycenter
            bcx = (xsum*np.arange(nx)).sum()/(xsum.sum())
            bcy = (ysum*np.arange(ny)).sum()/(ysum.sum())
            bcx_lst.append(bcx)
            bcy_lst.append(bcy)
        bcx_lst = np.array(bcx_lst)
        bcy_lst = np.array(bcy_lst)
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

        fig = plt.figure(figsize=(12, 5.5), dpi=200)
        ax = fig.add_axes([0.001, 0.52, 0.4, 0.4], projection=wcoord)
        ax.imshow(image, cmap='YlGnBu_r')
        _x1, _x2 = ax.get_xlim()
        _y1, _y2 = ax.get_ylim()

        # plot aperture
        bound_lst = get_aperture_bound(objmask)
        for (x1, y1, x2, y2) in bound_lst:
            ax.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-', lw=1)

        #mask = tictable['Tmag']<18
        #newtictable = tictable[mask]
        newtictable = tictable
        tmag_lst = newtictable['Tmag']
        ra_lst  = newtictable['RAJ2000']
        dec_lst = newtictable['DEJ2000']
        x_lst, y_lst = wcoord.all_world2pix(ra_lst, dec_lst, 0)
        size = np.maximum((18-tmag_lst)*10, 0.1)
        ax.scatter(x_lst, y_lst, s=size, c='none', ec='r', lw=0.5)

        ax.set_xlim(_x1, _x2)
        ax.set_ylim(_y1, _y2)

        xcoords = ax.coords[0]
        ycoords = ax.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.ticklabels.set_fontsize(7)
        ycoords.ticklabels.set_fontsize(7)
        xcoords.set_axislabel('RA (deg)',fontsize=7)
        ycoords.set_axislabel('Dec (deg)',fontsize=7)
        ax.grid(True, color='w', ls='--', lw=0.5)

        ################## plot pixel-by-pixel lc ######################
        #x0, y0 = wcoord.all_world2pix(ra, dec, 0)
        yy, xx = np.mgrid[:ny:, :nx:]
        y1 = max(min(yy[objmask])-2, 0)
        y2 = min(max(yy[objmask])+3, ny)
        x1 = max(min(xx[objmask])-2, 0)
        x2 = min(max(xx[objmask])+3, nx)
        flux_lst = {}
        for row in data1:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if (x, y) not in flux_lst:
                        flux_lst[(x,y)] = []
                    flux_lst[(x,y)].append(row['FLUX'][y,x])

        for y in range(y1, y2):
            for x in range(x1, x2):
                pixflux_lst = np.array(flux_lst[(x,y)])
                mask = ~np.isnan(pixflux_lst)
                med = np.median(pixflux_lst[mask])
                _w = 0.36/(x2-x1)
                _h = 0.4/(y2-y1)
                ax = fig.add_axes([0.33+(x-x1)*_w, 0.52+(y-y1)*_h, _w, _h])
                if objmask[y,x]:
                    color = 'C3'
                else:
                    color = 'C0'
                ax.plot(t_lst[mask], pixflux_lst[mask], lw=0.1, c=color)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
        ###################################

        fig.text(0.02, 0.48, '\n'.join(text_lst), fontsize=7,
                ha='left', va='top', fontfamily='monospace')


        ################## plot lc ############################
        axlc = fig.add_axes([0.33, 0.28, 0.36, 0.21])
        axlc.plot(tlc_lst, f_lst, 'o', c='C0', mew=0, alpha=0.5, ms=1)
        axlc.axvline(x=t_lst[idx], ls='--', c='k', lw=0.5)
        axlc.set_xlim(tlc_lst[0], tlc_lst[-1])
        for tick in axlc.xaxis.get_major_ticks():
            tick.label1.set_fontsize(7)
        for tick in axlc.yaxis.get_major_ticks():
            tick.label1.set_fontsize(7)
        axlc.set_xticklabels([])

        ################ plot barycen list #####################

        axbcx = fig.add_axes([0.33, 0.05, 0.36, 0.21])
        axbcy = axbcx.twinx()
        axbcx.plot(t_lst, bcx_lst, 'o', c='C1', mew=0, alpha=0.5, ms=1)
        axbcy.plot(t_lst, bcy_lst, 'o', c='C2', mew=0, alpha=0.5, ms=1)
        for axbc in [axbcx, axbcy]:
            axbc.set_xlim(tlc_lst[0], tlc_lst[-1])
        for tick in axbcx.xaxis.get_major_ticks():
            tick.label1.set_fontsize(7)
        for tick in axbcx.yaxis.get_major_ticks():
            tick.label1.set_fontsize(7)
            tick.label1.set_color('C1')
        for tick in axbcy.yaxis.get_major_ticks():
            tick.label2.set_fontsize(7)
            tick.label2.set_color('C2')
        axbcx.spines['left'].set_color('C1')
        axbcx.spines['right'].set_visible(False)
        axbcy.spines['right'].set_color('C2')
        axbcy.spines['left'].set_visible(False)
        axbcx.tick_params(axis='y', color='C1')
        axbcy.tick_params(axis='y', color='C2')
        axbcx.set_ylabel('X', fontsize=7, color='C1')
        axbcy.set_ylabel('Y', fontsize=7, color='C2')

        ############## plot large skyview ############
        wcoord2 = WCS(dss360head)

        ax2 = fig.add_axes([0.67, 0.52, 0.4, 0.4], projection=wcoord2)
        ax2.imshow(dss360data, cmap='gray_r')

        # plot pixel grid
        for iy in np.arange(-0.5, ny, 1):
            x_lst = [-0.5, nx-0.5]
            y_lst = [iy, iy]
            ra_lst, dec_lst = wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst, '-', c='b', lw=0.3)
        for ix in np.arange(-0.5, nx, 1):
            x_lst = [ix, ix]
            y_lst = [-0.5, ny-0.5]
            ra_lst, dec_lst = wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst, '-', c='b', lw=0.3)

        # plot aperture
        bound_lst = get_aperture_bound(objmask)
        for (x1, y1, x2, y2) in bound_lst:
            ra_lst, dec_lst = wcoord.all_pix2world(
                    [x1-0.5,x2-0.5], [y1-0.5,y2-0.5], 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            ax2.plot(x2_lst, y2_lst,  'r-', lw=0.7)
        
        # plot x and y arrows
        for x_lst, y_lst in [([-1.0, +1.5], [-1.0, -1.0]),
                             ([-1.0, -1.0], [-1.0, +1.5])]:
            ra_lst, dec_lst = wcoord.all_pix2world(x_lst, y_lst, 0)
            x2_lst, y2_lst = wcoord2.all_world2pix(ra_lst, dec_lst, 0)
            x, dx = x2_lst[0], x2_lst[1]-x2_lst[0]
            y, dy = y2_lst[0], y2_lst[1]-y2_lst[0]
            ax2.arrow(x, y, dx, dy, width=1, color='k', lw=0, head_width=5,
                    head_length=10)
        _ra, _dec = wcoord.all_pix2world(2.5, -1, 0)
        _x, _y = wcoord2.all_world2pix(_ra, _dec, 0)
        ax2.text(_x, _y, 'x', fontsize=7, ha='center', va='center')
        _ra, _dec = wcoord.all_pix2world(-1, 2.5, 0)
        _x, _y = wcoord2.all_world2pix(_ra, _dec, 0)
        ax2.text(_x, _y, 'y', fontsize=7, ha='center', va='center')

        xcoords = ax2.coords[0]
        ycoords = ax2.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.ticklabels.set_fontsize(7)
        ycoords.ticklabels.set_fontsize(7)
        xcoords.set_axislabel('RA (deg)',fontsize=7)
        ycoords.set_axislabel('Dec (deg)',fontsize=7)

        ################## plot small skyview ###########################
        wcoord3 = WCS(dss100head)

        ax3 = fig.add_axes([0.67, 0.05, 0.4, 0.4], projection=wcoord3)
        ax3.imshow(dss100data, cmap='gray_r')

        _x1, _x2 = ax3.get_xlim()
        _y1, _y2 = ax3.get_ylim()

        #### plot circles for different separations
        x0, y0 = wcoord3.all_world2pix(coord.ra.deg, coord.dec.deg, 0)
        for rc in np.arange(10, 60+1, 10):
            _ra_lst, _dec_lst = get_circle(coord.ra.deg, coord.dec.deg, rc/3600)
            x1_lst, y1_lst = wcoord3.all_world2pix(_ra_lst, _dec_lst, 0)
            ax3.plot(x1_lst, y1_lst, '--', color='C2', lw=0.5)
            if y1_lst[0]<_y2:
                ax3.text(x1_lst[0], y1_lst[0], '{:}"'.format(rc),
                        ha='center', color='C2', fontsize=6)

        #### plot nearby stars and proper motion arrows ######
        pmlen = 1000
        m = gaiae3table['Gmag']<18

        # bright stars
        ra_lst = gaiae3table[m]['RAJ2000']
        dec_lst = gaiae3table[m]['DEJ2000']
        x_lst, y_lst = wcoord3.all_world2pix(ra_lst, dec_lst, 0)
        ax3.plot(x_lst, y_lst, 'o', ms=5, color='none', mec='C0', mew=0.8)
        # dark stars
        ra_lst = gaiae3table[~m]['RAJ2000']
        dec_lst = gaiae3table[~m]['DEJ2000']
        x_lst, y_lst = wcoord3.all_world2pix(ra_lst, dec_lst, 0)
        ax3.plot(x_lst, y_lst, 'o', ms=0.5, color='none', mec='C0', mew=0.8)
        # plot proper motion for ALL stars
        ra_lst = gaiae3table['RAJ2000']
        dec_lst = gaiae3table['DEJ2000']
        x_lst, y_lst = wcoord3.all_world2pix(ra_lst, dec_lst, 0)
        pmra_lst  = gaiae3table['pmRA']
        pmdec_lst = gaiae3table['pmDE']
        d_ra = pmra_lst*1e-3/3600./np.cos(np.deg2rad(dec_lst))*pmlen
        d_dec = pmdec_lst*1e-3/3600.*pmlen
        ra2_lst = ra_lst + d_ra
        dec2_lst = dec_lst + d_dec
        x2_lst, y2_lst = wcoord3.all_world2pix(ra2_lst, dec2_lst, 0)
        dx_lst = x2_lst - x_lst
        dy_lst = y2_lst - y_lst
        ax3.quiver(x_lst, y_lst, dx_lst, dy_lst, width=2, units='dots',
                 angles='xy', scale_units='xy', scale=1, color='C0')
        ax3.set_xlim(_x1, _x2)
        ax3.set_ylim(_y1, _y2)
        #ax3.grid(True, ls='--', lw=0.5)

        xcoords = ax3.coords[0]
        ycoords = ax3.coords[1]
        xcoords.set_major_formatter('d.ddd')
        ycoords.set_major_formatter('d.ddd')
        xcoords.ticklabels.set_fontsize(7)
        ycoords.ticklabels.set_fontsize(7)
        #xcoords.set_axislabel('RA (deg)',fontsize=7)
        ycoords.set_axislabel('Dec (deg)',fontsize=7)

        #plt.show()
        title = 'TIC {} (Sector {})'.format(tic, sector)
        fig.suptitle(title)
        fig.savefig(figname)
        plt.close(fig)

