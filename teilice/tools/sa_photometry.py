import re
import os
import io
import sys
import time
import argparse

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.io.fits as fits
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.ticker as tck

from astroquery.vizier import Vizier

import tkinter as tk
import tkinter.ttk as ttk

from .. import TessTarget
from ..utils import get_lc_sectors
from ..visual import get_aperture_bound

def sector_string_to_lst(sector_string):
    sector_lst = []
    for v in sector_string.split(','):
        v = v.strip()
        if '-' in v:
            group = v.split('-')
            s1 = int(group[0])
            s2 = int(group[-1])
            for s in range(s1, s2+1):
                sector_lst.append(int(s))
        elif v.isdigit():
            sector_lst.append(int(v))
        else:
            continue
    return sector_lst

def sector_lst_to_string(sector_lst):
    group_lst = np.split(sector_lst, np.where(np.diff(sector_lst) > 1)[0]+1)
    string_lst = []
    for group in group_lst:
        if len(group)>2:
            string_lst.append('{}-{}'.format(group[0], group[-1]))
        else:
            for s in group:
                string_lst.append(str(s))
    return ','.join(string_lst)


def find_saved_sa_files(main_tic, tic):
    path = os.path.join(datapool, 'sa-lc')
    
    sector_lst = []
    for sector_path in sorted(os.listdir(path)):
        mobj = re.match('s(\d+)', sector_path)
        if not mobj:
            continue
        path2 = os.path.join(path, sector_path)
        sector = int(mobj.group(1))

        for fname in os.listdir(path2):
            if not fname.endswith('.fits'):
                continue
            col = fname.split('-')
            _tic      = int(col[1])
            _main_tic = int(col[2])
            if _tic == tic and _main_tic == main_tic:
                sector_lst.append(sector)
                break
    return sector_lst

class MainWindow(tk.Frame):

    def __init__(self, master, width, height, source_filename):
        self.master = master

        tk.Frame.__init__(self, master, width=width, height=height)


        # set target information
        self.tic = -1
        self.sector = -1
        self.selected_tic = -1

        self.set_nearby = False

        right_width = 500

        self.plot_frame = PlotFrame(master=self,
                                    width = width - right_width,
                                    height = height - 300,
                                    )
        
        self.right_frame = RightFrame(master = self,
                                        width  = right_width,
                                        height = height,
                                        source_filename = source_filename,
                                        )
        self.plot_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.pack()


    def load_nearby_table(self, tic):
        """
        """
        
        t1 = time.time()
        # read tic tables
        cache_tic = os.path.join(datapool, 'cache_ticv82',
                '{:02d}'.format(tic%100))
        ticfilename = os.path.join(cache_tic,
                'tic_nearby_{:012d}_250.vot'.format(tic))
        if os.path.exists(ticfilename):
            tictable = Table.read(ticfilename)
        else:
            # query new nearby table
            catid = 'IV/39/tic82'

            # query target coordinate
            tablelist = Vizier(catalog=catid, columns=['**'],
                        column_filters={'TIC': '={}'.format(tic)}
                        ).query_constraints()
            ticrow = tablelist[catid][0]
            coord = SkyCoord(ticrow['RAJ2000'], ticrow['DEJ2000'], unit='deg')

            viz = Vizier(catalog=catid, columns=['**', '+_r'])
            viz.ROW_LIMIT = -1
            r = 250
            tablelist = viz.query_region(coord, radius=r*u.arcsec)
            tictable = tablelist[catid]

            # save it
            tictable.write(ticfilename, format='votable', overwrite=True)

        self.tictable = tictable

        self.coord_lst = SkyCoord(self.tictable['RAJ2000'],
                                  self.tictable['DEJ2000'], unit='deg')
        
        # read gaia2 tables
        #cache_gaia2 = os.path.join(datapool, 'cache_gaia2',
        #        '{:02d}'.format(tic%100))
        #gaia2filename = os.path.join(cache_gaia2,
        #        'gaia2_nearby_{:012d}_150.vot'.format(tic))
        #if os.path.exists(gaia2filename):
        #    gaia2table = Table.read(gaia2filename)
        #else:
        #    pass
        
        # read gaia3 tables
        cache_gaia3 = os.path.join(datapool, 'cache_gaia3',
                '{:02d}'.format(tic%100))
        #gaia3filename = os.path.join(cache_gaia3,
        #        'gaia3_nearby_{:012d}_150.vot'.format(tic))
        #if os.path.exists(gaia3filename):
        #    gaia3table = Table.read(gaia3filename)
        #else:
        #    pass
        gaia3varfilename = os.path.join(cache_gaia3,
                'gaia3var_nearby_{:012d}_180.vot'.format(tic))
        if os.path.exists(gaia3varfilename):
            gaia3vartable = Table.read(gaia3varfilename)
        else:
            catid = 'I/358/vclassre'
            viz = Vizier(catalog=catid, columns=['**', '+_r'])
            viz.ROW_LIMIT = -1
            target = TessTarget(tic)
            tablelist = viz.query_region(target.coord, radius=180*u.arcsec)
            if len(tablelist)>0:
                gaia3vartable = tablelist[catid]
            else:
                # generate an empty table
                viz = Vizier(catalog=catid, columns=['**'])
                viz.ROW_LIMIT = 10
                tablelist = viz.query_constraints()
                tab = tablelist[catid]
                m = [False]*len(tab)
                gaia3vartable = tab[m]
            gaia3vartable.write(gaia3varfilename, format='votable', overwrite=True)
        self.gaia3vartable = gaia3vartable
        
        mixed_table1, mixed_table2 = get_mixed_table(
                tic, tictable,
                #gaia2table, gaia3table,
                gaia3vartable)

        self.plot_frame.nearby_frame.load_mixed_table(
                        mixed_table1,
                        mixed_table2,
                        main_tic = tic)
        t4 = time.time()
        #print(t4 - t1, 'mixing tables')

        self.set_nearby = True


    def plot(self, tic, sector):

        if tic == self.tic and sector == self.sector:
            # do not need to replot
            return None
        

        self.tic    = tic
        self.sector = sector
        target = TessTarget(tic)


        ### redraw
        lcdata = target.get_lc(sector, datapool=datapool)
        tpdata = target.get_tp(sector, datapool=datapool)
        self.lcdata = lcdata

        ########### read tp file #############
        #t_lst, imgq_lst, image_lst, _, wcoord = read_tp(image_file)
        t_lst, imgq_lst, image_lst, _, wcoord = tpdata
        self.wcoord1 = wcoord
        self.image_lst = image_lst
        nlayer, ny, nx = image_lst.shape

        # mnan is to find the images that contains real-number pixels
        # i.e., not all of the pixels in this frame is NaN
        mnan = np.array([(~np.isnan(image_lst[i])).sum()>0 for i in np.arange(nlayer)])
        m2 = (imgq_lst==0) * mnan

        # determine best frame to be displayed
        m1 = (lcdata.q_lst==0)*(~np.isnan(lcdata.t_lst))
        # find the median flux value
        medf = np.median(lcdata.flux_lst[m1])
        # find the index of the frame that closest to the median flux value
        idx1 = np.abs(lcdata.flux_lst[m1] - medf).argmin()
        # find the time of the frame that closest to the median flux value
        t = lcdata.t_lst[m1][idx1]
        # on lcdata, find the index of the frame that closes to the time of
        # median flux
        idx = np.abs(t_lst[m2] - t).argmin()
        image = image_lst[m2][idx]
        ny, nx = image.shape


        fig = self.plot_frame.fig
        fig.clear()

        # set title
        title_text = 'FoV of TIC {} (Sector {})'.format(self.tic, self.sector)
        self.title = fig.suptitle(title_text, fontsize=9)
        # add axes
        ax = fig.add_axes([-0.20, 0.1, 0.8, 0.8], projection=wcoord)
        self.axtp = ax
        ax.imshow(image, cmap='YlGnBu_r')
        _x1, _x2 = ax.get_xlim()
        _y1, _y2 = ax.get_ylim()

        # plot official aperture using red
        custmap = mcolors.LinearSegmentedColormap.from_list('TransRed',
                    [(1,0,0,0), (1,0,0,0.3)], N=2)
        aperture = lcdata.imagemask & 2 >0
        bound_lst = get_aperture_bound(aperture)
        for (x1, y1, x2, y2) in bound_lst:
            ax.plot([x1-0.5, x2-0.5], [y1-0.5, y2-0.5], 'r-', lw=0.8)
        #ax.imshow(aperture, cmap=custmap)

        # plot customized aperture using red dashed
        #line2d, = ax.plot([-1,-1],[-1,-1],'r--', lw=0.8)
        #self.custom_bounds_line2d = line2d
        self.custom_apertures = np.zeros((ny, nx), dtype=bool)
        self.custom_img = ax.imshow(self.custom_apertures, cmap=custmap, vmin=0, vmax=1)


        t22 = time.time()
        #print(t22 - t2)
        t4 = time.time()


        # plot nearby sources
        tictable = self.tictable
        tic_lst  = tictable['TIC']
        tmag_lst = tictable['Tmag']
        ra_lst   = tictable['RAJ2000']
        dec_lst  = tictable['DEJ2000']
        x_lst, y_lst = wcoord.all_world2pix(ra_lst, dec_lst, 0)
        size = np.maximum((18-tmag_lst)*10, 0.1)
        # plot target star and other stars with different linewidths
        #self.x_lst = x_lst
        #self.y_lst = y_lst
        #for x, y in zip(self.x_lst, self.y_lst):
        #        print(x, y)
        self.scatters = ax.scatter(x_lst, y_lst, s=size, c='none', ec='r', lw=0.3)
        m = tic_lst==target.tic
        linewidths = np.ones(len(tictable))*0.3
        linewidths[m] = 1
        self.scatters.set_linewidths(linewidths)

        # plot gaia3 variables
        ra_lst  = self.gaia3vartable['RA_ICRS']
        dec_lst = self.gaia3vartable['DE_ICRS']
        x_lst, y_lst = wcoord.all_world2pix(ra_lst, dec_lst, 0)
        ax.plot(x_lst, y_lst, 'o', c='C2', ms=2, mew=0)


        ax.set_xlim(_x1, _x2)
        ax.set_ylim(_y1, _y2)

        xcoords = ax.coords[0]
        ycoords = ax.coords[1]
        xcoords.set_major_formatter('d.dd')
        ycoords.set_major_formatter('d.dd')
        xcoords.ticklabels.set_fontsize(7)
        ycoords.ticklabels.set_fontsize(7)
        xcoords.set_axislabel('RA (deg)',fontsize=7)
        ycoords.set_axislabel('Dec (deg)',fontsize=7)
        ax.grid(True, color='w', ls='--', lw=0.3)
        t5 = time.time()
        #print(t5 - t4, 'plotting nearby stars')


        ################## plot pixel-by-pixel lc ######################

        self.pbp_axes = {}
        #x0, y0 = wcoord.all_world2pix(ra, dec, 0)
        yy, xx = np.mgrid[:ny:, :nx:]
        #y1 = max(min(yy[aperture])-2, 0)
        #y2 = min(max(yy[aperture])+3, ny)
        #x1 = max(min(xx[aperture])-2, 0)
        #x2 = min(max(xx[aperture])+3, nx)
        y1, y2 = 0, ny
        x1, x2 = 0, nx
        flux_lst = {}
        for image in image_lst[m2]:
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if (x, y) not in flux_lst:
                        flux_lst[(x,y)] = []
                    flux_lst[(x,y)].append(image[y,x])
        
        for y in range(y1, y2):
            for x in range(x1, x2):
                pixflux_lst = np.array(flux_lst[(x,y)])
                mask = ~np.isnan(pixflux_lst)
                if mask.sum()==0:
                    continue
                _w = 0.35/(x2-x1)
                _h = 0.8/(y2-y1)
                ax = fig.add_axes([0.35+(x-x1)*_w, 0.1+(y-y1)*_h, _w, _h])
                self.pbp_axes[(y,x)] = ax
                if aperture[y,x]:
                    color = 'C3'
                else:
                    color = 'C0'
                # divide light curve from sector gap
                xplot_lst = t_lst[m2][mask]
                yplot_lst = pixflux_lst[mask]
                idx = np.diff(xplot_lst).argmax()+1
                ax.plot(xplot_lst[0:idx], yplot_lst[0:idx], lw=0.1, c=color)
                ax.plot(xplot_lst[idx:],  yplot_lst[idx:],  lw=0.1, c=color)

                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                # set line widths of axis
                ax.spines['left'].set_linewidth(0.5)
                ax.spines['right'].set_linewidth(0.5)
                ax.spines['top'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)
        
        t6 = time.time()
        #print(t6 - t5)

        ################ plot LC #############
        axlc1 = fig.add_axes([0.725, 0.53, 0.25, 0.37])
        axlc2 = fig.add_axes([0.725, 0.10, 0.25, 0.37])
        self.axlc1 = axlc1
        self.axlc2 = axlc2
        
        m1 = (lcdata.q_lst==0) * (~np.isnan(lcdata.t_lst))
        self.m1 = m1
        axlc1.plot(lcdata.t_lst[m1], lcdata.flux_lst[m1], 'o',
                c='C0', mew=0, ms=0.6)

        self.custom_flux = np.zeros_like(lcdata.flux_lst)
        self.custom_line, = axlc2.plot(lcdata.t_lst[m1], self.custom_flux[m1], 'o',
                c='C3', mew=0, ms=0.6)

        axlc1.grid(True, ls='--', lw=0.5, c='gray', alpha=0.3)
        axlc1.set_axisbelow(True)
        axlc2.grid(True, ls='--', lw=0.5, c='gray', alpha=0.3)
        axlc2.set_axisbelow(True)
        axlc1.xaxis.set_major_locator(tck.MultipleLocator(5))
        axlc1.xaxis.set_minor_locator(tck.MultipleLocator(1))
        axlc2.xaxis.set_major_locator(tck.MultipleLocator(5))
        axlc2.xaxis.set_minor_locator(tck.MultipleLocator(1))
        axlc1.set_xlim(lcdata.t_lst[m1][0], lcdata.t_lst[m1][-1])
        axlc2.set_xlim(lcdata.t_lst[m1][0], lcdata.t_lst[m1][-1])
        axlc2.set_xlabel('Time (BJD-2457000)', fontsize=7)

        # draw secondary Y axis in the right
        axlc1c = axlc1.twinx()
        med1 = np.median(lcdata.flux_lst[m1])
        _y1, _y2 = axlc1.get_ylim()
        axlc1c.set_ylim(_y1/med1, _y2/med1)


        # set text in axlc2
        _x1, _x2 = axlc2.get_xlim()
        _y1, _y2 = axlc2.get_ylim()
        self.axlc2text = axlc2.text(0.95*_x1+0.05*_x2, 0.1*_y1+0.9*_y2, '', fontsize=7)
        axlc2.set_xlim(_x1, _x2)
        axlc2.set_ylim(_y1, _y2)


        for tick in axlc1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in axlc1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in axlc2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        for tick in axlc2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(6)
        # set fontsize in ticks of secondary Y axis
        for tick in axlc1c.yaxis.get_major_ticks():
            tick.label2.set_fontsize(6)


        # plot existing data of selected tic
        #if selected_tic is not None and selected_tic != tic:
        if False:
            fname = 'tess-{:012d}-{:012d}-s{:03d}-s_lc.fits'.format(
                        selected_tic, tic, sector)
            path = os.path.join(datapool, 'sa-lc', 's{:03d}'.format(sector))

            filename = os.path.join(path, fname)
            if os.path.exists(filename):
                hdulst = fits.open(filename)
                lc_table = hdulst[1].data
                mask     = hdulst[2].data
                hdulst.close()

                # set aperture plot in the left TP axes
                self.custom_apertures = mask & 2 > 0
                self.custom_img.set(data=self.custom_apertures)
                
                # set aperture plot in the center PBP axes
                for (y, x), ax in self.pbp_axes.items():
                    if self.custom_apertures[y, x]:
                        ax.patch.set_facecolor('#ffbbdd')
                    else:
                        ax.patch.set_facecolor('w')

                # set LC
                flux_lst = lc_table['SAP_FLUX']
                ydata = flux_lst[self.m1]
                self.custom_line.set_ydata(ydata)

                # reset xlim and ylim
                vmin = np.nanmin(ydata)
                vmax = np.nanmax(ydata)
                vspan = vmax-vmin
                self.axlc2.set_ylim(vmin-vspan*0.1, vmax+vspan*0.1)

                # reset text of axlc2
                self.axlc2text.set_text('TIC {}'.format(selected_tic))
                # reset position of axlc2
                _x1, _x2 = self.axlc2.get_xlim()
                _y1, _y2 = self.axlc2.get_ylim()
                self.axlc2text.set_x(0.95*_x1+0.05*_x2)
                self.axlc2text.set_y(0.1*_y1+0.9*_y2)


        self.plot_frame.focus_set()


    def reset_custom_lc(self):
        # reset to zero
        self.custom_apertures = np.zeros_like(self.image_lst[0], dtype=bool)
        self.custom_img.set(data=self.custom_apertures)

        # set aperture plot in the center PBP axes
        for (y, x), ax in self.pbp_axes.items():
            ax.patch.set_facecolor('w')

        self.custom_flux = np.zeros_like(self.lcdata.flux_lst)
        self.custom_line.set_ydata(self.custom_flux[self.m1])
        self.axlc2text.set_text('')
        # update state of saving lc button
        self.right_frame.save_lc_button['state'] = tk.DISABLED

    def plot_selected_tic(self, tic, sector, selected_tic):

        fname = 'tess-{:012d}-{:012d}-s{:03d}-s_lc.fits'.format(
                    selected_tic, tic, sector)
        path = os.path.join(datapool, 'sa-lc', 's{:03d}'.format(sector))

        filename = os.path.join(path, fname)
        if os.path.exists(filename):
            hdulst = fits.open(filename)
            lc_table = hdulst[1].data
            mask     = hdulst[2].data
            hdulst.close()

            # set aperture plot in the left TP axes
            self.custom_apertures = mask & 2 > 0
            self.custom_img.set(data=self.custom_apertures)
            
            # set aperture plot in the center PBP axes
            for (y, x), ax in self.pbp_axes.items():
                if self.custom_apertures[y, x]:
                    ax.patch.set_facecolor('#ffbbdd')
                else:
                    ax.patch.set_facecolor('w')

            # set LC
            flux_lst = lc_table['SAP_FLUX']
            self.custom_flux = flux_lst
            ydata = flux_lst[self.m1]
            self.custom_line.set_ydata(ydata)

            # reset xlim and ylim
            vmin = np.nanmin(ydata)
            vmax = np.nanmax(ydata)
            vspan = vmax-vmin
            self.axlc2.set_ylim(vmin-vspan*0.1, vmax+vspan*0.1)

            # reset text of axlc2
            self.axlc2text.set_text('TIC {}'.format(selected_tic))
            # reset position of axlc2
            _x1, _x2 = self.axlc2.get_xlim()
            _y1, _y2 = self.axlc2.get_ylim()
            self.axlc2text.set_x(0.95*_x1+0.05*_x2)
            self.axlc2text.set_y(0.1*_y1+0.9*_y2)

            self.right_frame.clearap_button['state'] = tk.NORMAL


    def extract_lc(self):
        if self.custom_apertures.sum()>0:
            nlayer = self.image_lst.shape[0]
            for ilayer in range(nlayer):
                flux = (self.image_lst[ilayer][self.custom_apertures]).sum()
                self.custom_flux[ilayer] = flux
            ydata = self.custom_flux[self.m1]
            self.custom_line.set_ydata(ydata)

            # reset xlim and ylim
            vmin = np.nanmin(ydata)
            vmax = np.nanmax(ydata)
            vspan = vmax-vmin
            self.axlc2.set_ylim(vmin-vspan*0.1, vmax+vspan*0.1)

            ## get selected TIC number
            selected_tic = self.plot_frame.nearby_frame.get_selected_tic()

            # reset text of axlc2
            self.axlc2text.set_text('TIC {}'.format(selected_tic))
            # reset position of axlc2
            _x1, _x2 = self.axlc2.get_xlim()
            _y1, _y2 = self.axlc2.get_ylim()
            self.axlc2text.set_x(0.95*_x1+0.05*_x2)
            self.axlc2text.set_y(0.1*_y1+0.9*_y2)

            # update state of saving lc button
            self.right_frame.save_lc_button['state'] = tk.NORMAL
            # udpate state of clear aperture button
            self.right_frame.clearap_button['state'] = tk.NORMAL
        else:
            # no light curve
            self.custom_flux = np.zeros_like(self.lcdata.flux_lst)
            self.custom_line.set_ydata(self.custom_flux[self.m1])

            self.axlc2text.set_text('')
            # update state of saving lc button
            self.right_frame.save_lc_button['state'] = tk.DISABLED
            # udpate state of clear aperture button
            self.right_frame.clearap_button['state'] = tk.DISABLED



class SectorTable(tk.Frame):
    def __init__(self, master, width, height):
        self.master = master
        tk.Frame.__init__(self, master, width=width, height=height)

        self.sector_tree = ttk.Treeview(
                master  = self,
                columns = ('Sector', 'cadence', 'comments'),
                show    = 'headings',
                style   = 'Treeview',
                height  = 5,
                selectmode = 'browse',
                )
        self.sector_tree.bind('<<TreeviewSelect>>', self.on_select_item)

        self.scrollbar = tk.Scrollbar(master=self,
                orient = tk.VERTICAL,
                width  = 10,
                )
        self.sector_tree.column('Sector',   width=80, anchor='e')
        self.sector_tree.column('cadence',  width=80, anchor='e')
        self.sector_tree.column('comments', width=width-180)
        self.sector_tree.heading('Sector',text='Sector')
        self.sector_tree.heading('cadence',text='cadence')
        self.sector_tree.heading('comments',text='')
        self.sector_tree.config(yscrollcommand = self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=30)

        self.scrollbar.config(command=self.sector_tree.yview)

        
        self.sector_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        self.pack()

    def load_sectors(self, sector_lst, idx=0):

        for item in self.sector_tree.get_children():
            self.sector_tree.delete(item)

        for i, sector in enumerate(sector_lst):
            item = (sector, 120, '')
            iid = self.sector_tree.insert('', tk.END,
                    values=item, tags='normal')
            if i==idx:
                # this will trigger on_select_item function, then
                # trigger plot function
                self.sector_tree.selection_set(iid)

    def on_select_item(self, event):
        """event handler for clicking an item in sector table.
        """
        items = event.widget.selection()
        if len(items)>0:


            # set aliases
            main_window = self.master.master
            nearby_frame = main_window.plot_frame.nearby_frame
            source_tree = self.master.source_frame.source_tree

            # get current item in source table
            item = source_tree.selection()[0]
            values = source_tree.item(item, 'values')
            tic = int(values[0])
            if len(values[1])==0:
                # selected tic is an orginal TIC
                selected_tic = tic
            else:
                # selected tic is not an origional TIC
                selected_tic = int(values[1])


            # get the clicked sector number
            item = items[0]
            values = self.sector_tree.item(item, 'values')
            sector = int(values[0])
        
            # will choose the selected target here
            if self.master.master.set_nearby:
                # refresh for same tic but different sector
                # nothing to do in this case
                pass
            else:
                # refresh from new source
                main_window.load_nearby_table(tic)
            nearby_frame.select_tic(selected_tic)
        
            # will trigger plotting function here
            self.master.master.plot(tic, sector)
            if selected_tic != tic:
                main_window.plot_selected_tic(tic, sector, selected_tic)
            
            main_window.plot_frame.canvas.draw()


class PlotFigure(Figure):

    def __init__(self, width, height):
        dpi=200
        figsize  = (width/dpi, height/dpi)
        super(PlotFigure, self).__init__(figsize=figsize, dpi=dpi)
        #self.patch.set_facecolor('w')


class PlotFrame(tk.Frame):
    def __init__(self, master, width, height):
        self.master = master
        tk.Frame.__init__(self, master, width=width, height=height)

        self.fig = PlotFigure(width = width,
                              height = height,
                              )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.start_xy = None


        self.nearby_frame = NearbyTable(master=self,
                            width  = width,
                            height = 280,
                            )
        self.nearby_frame.pack(side=tk.TOP, pady=5)

        self.pack()

    def on_click(self, event):
        """event handler for cliking in the figure.
        """

        if not hasattr(self.master, 'axtp'):
            return

        if event.inaxes==self.master.axtp:
            # select a source in the TP figure

            threshold = 0.03

            # get all coordiantes of scatters
            xydata = self.master.scatters.get_offsets()
            x_lst = xydata[:, 0]
            y_lst = xydata[:, 1]
            # calculate disstances
            dist_lst = (x_lst - event.xdata)**2 + \
                       (y_lst - event.ydata)**2
            mini = dist_lst.argmin()

            if dist_lst.min() <= threshold:
                _row = self.master.tictable[mini]
                _tic = _row['TIC']

                for ii, item in enumerate(self.nearby_frame.nearby_tree.get_children()):
                    values = self.nearby_frame.nearby_tree.item(item, 'values')
                    if int(values[0])==_tic:
                        self.nearby_frame.nearby_tree.selection_set(item)
                        self.nearby_frame.nearby_tree.yview(ii)
                        break
        else:
            # select aperture mode


            for (y,x), ax in self.master.pbp_axes.items():
                if event.inaxes==ax:
                    #if event.dblclick:
                    #    # in multi-selection mode
                    #    if self.start_xy is None:
                    #        self.start_xy = (x, y)
                    #    else:
                    #        x0, y0 = self.start_xy
                    #        x1 = min(x0, x)
                    #        x2 = max(x0, x)
                    #        y1 = min(y0, y)
                    #        y2 = max(y0, y)
                    #        print(x0, y0, x, y)
                    #        for _x in range(x1, x2+1):
                    #            for _y in range(y1, y2+1):
                    #                self.master.custom_apertures[_y, _x] = ~self.master.custom_apertures[_y, _x]
                    #        self.start_xy = None
                    #    if self.master.custom_apertures[y, x]:
                    #        ax.patch.set_facecolor('#ffbbdd')
                    #    else:
                    #        ax.patch.set_facecolor('w')
                    #else:
                    #    self.start_xy = None

                    # in single-selection mode
                    self.master.custom_apertures[y, x] = ~self.master.custom_apertures[y, x]
                    if self.master.custom_apertures[y, x]:
                        ax.patch.set_facecolor('#ffbbdd')
                    else:
                        ax.patch.set_facecolor('w')
                    ##newly added
                    break

            self.master.custom_img.set(data=self.master.custom_apertures)

            # re-extract light curves
            self.master.extract_lc()
            self.canvas.draw()
                        


class NearbyTable(tk.Frame):
    def __init__(self, master, width, height):

        self.master = master
        tk.Frame.__init__(self, master, width=width, height=height)


        columns = ('TIC', 'm_TIC', 'Disp', 'r', 'Tmag', 'Gaia2', 'Gaia3',
                'VarClass', 'Period', 'action')
        self.nearby_tree = ttk.Treeview(
                master = self,
                columns = columns,
                show    = 'headings',
                style   = 'Treeview',
                height  = 8,
                selectmode = 'browse',
                )
        self.nearby_tree.bind('<<TreeviewSelect>>', self.on_select_bright_tree)
        self.nearby_tree.bind('<Button-1>', self.on_click_neb)

        self.scrollbar = tk.Scrollbar(master = self,
                orient = tk.VERTICAL,
                width  = 10,
                )
        self.nearby_tree.column('TIC', width=90, anchor='e')
        self.nearby_tree.column('m_TIC', width=90, anchor='e')
        self.nearby_tree.column('Disp', width=80)
        self.nearby_tree.column('r', width=60, anchor='e')
        self.nearby_tree.column('Tmag', width=60, anchor='e')
        self.nearby_tree.column('Gaia2', width=180, anchor='e')
        self.nearby_tree.column('Gaia3', width=180, anchor='e')
        self.nearby_tree.column('VarClass', width=140)
        self.nearby_tree.column('Period', width=80)
        self.nearby_tree.column('action', width=100)

        self.nearby_tree.heading('TIC', text='TIC')
        self.nearby_tree.heading('m_TIC', text='m_TIC')
        self.nearby_tree.heading('Disp',  text='Disp')
        self.nearby_tree.heading('r',  text='r')
        self.nearby_tree.heading('Tmag',  text='Tmag')
        self.nearby_tree.heading('Gaia2',  text='Gaia DR2')
        self.nearby_tree.heading('Gaia3',  text='Gaia DR3')
        self.nearby_tree.heading('VarClass', text='VarClass')
        self.nearby_tree.heading('Period', text='Period')
        self.nearby_tree.heading('action', text='Action')

        self.nearby_tree.config(yscrollcommand = self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=25)

        self.scrollbar.config(command=self.nearby_tree.yview)


        self.nearby_tree2 = ttk.Treeview(
                master  = self,
                columns = columns,
                show    = 'headings',
                style   = 'Treeview',
                height  = 8,
                selectmode = 'browse',
                )
        self.nearby_tree2.bind('<<TreeviewSelect>>', self.on_select_fiant_tree)
        self.nearby_tree2.bind('<Button-1>', self.on_click_neb)

        self.scrollbar2 = tk.Scrollbar(master = self,
                orient = tk.VERTICAL,
                width  = 10,
                )
        self.nearby_tree2.column('TIC', width=90, anchor='e')
        self.nearby_tree2.column('m_TIC', width=90, anchor='e')
        self.nearby_tree2.column('Disp', width=80)
        self.nearby_tree2.column('r',  width=60, anchor='e')
        self.nearby_tree2.column('Tmag', width=60, anchor='e')
        self.nearby_tree2.column('Gaia2', width=180, anchor='e')
        self.nearby_tree2.column('Gaia3', width=180, anchor='e')
        self.nearby_tree2.column('VarClass', width=140)
        self.nearby_tree2.column('Period', width=80)
        self.nearby_tree2.column('action', width=100)

        self.nearby_tree2.heading('TIC', text='TIC')
        self.nearby_tree2.heading('m_TIC', text='m_TIC')
        self.nearby_tree2.heading('Disp',  text='Disp')
        self.nearby_tree2.heading('r',  text='r')
        self.nearby_tree2.heading('Tmag',  text='Tmag')
        self.nearby_tree2.heading('Gaia2',  text='Gaia DR2')
        self.nearby_tree2.heading('Gaia3',  text='Gaia DR3')
        self.nearby_tree2.heading('VarClass', text='VarClass')
        self.nearby_tree2.heading('Period', text='Period')
        self.nearby_tree2.heading('action', text='Action')

        self.nearby_tree2.config(yscrollcommand = self.scrollbar2.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=25)

        self.scrollbar2.config(command=self.nearby_tree2.yview)

        self.nearby_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.nearby_tree2.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.scrollbar2.pack(side=tk.LEFT, fill=tk.Y)
        self.pack()


    def load_mixed_table(self, tab1, tab2, main_tic):

        # clear selection in bright catalogue
        # this will trigger the on_select_bright_tree() if there is a selection
        for item in self.nearby_tree.get_children():
            self.nearby_tree.delete(item)

        # get already added TICs and put them in existing_tic_lst
        source_tree = self.master.master.right_frame.source_frame.source_tree
        existing_tic_lst = []
        for item in source_tree.get_children():
            values = source_tree.item(item, 'value')
            if int(values[0])==main_tic:
                for item2 in source_tree.get_children(item):
                    values2 = source_tree.item(item2, 'value')
                    if len(values2[1])>0:
                        tic = int(values2[1])
                        existing_tic_lst.append(tic)
                break

        for row in tab1:
            if row['TIC']==main_tic:
                action = ''
            elif row['TIC'] in existing_tic_lst:
                action = 'Added'
            else:
                action = 'Add New'
            item = (row['TIC'], row['m_TIC'], row['Disp'],
                    '{:6.2f}'.format(row['_r']),
                    '' if row['Tmag'] is np.ma.masked else '{:6.3f}'.format(row['Tmag']),
                    row['GAIA'],
                    row['Gaia3'] if row['Gaia3']>0 else '',
                    row['VarClass'],
                    '{:8.4f}'.format(row['Period']) if not np.isnan(row['Period']) else '',
                    action)
            iid = self.nearby_tree.insert('', tk.END, values=item, tags='normal')
        # add an empty TIC to this list
        self.nearby_tree.insert('', tk.END, values=(0, '--', '',
                        '', '', '', '', '', '', 'Add New'), tags='normal')

        # clear selection in faint catalogue
        for item in self.nearby_tree2.get_children():
            self.nearby_tree2.delete(item)

        for row in tab2:
            if row['TIC']==main_tic:
                action = ''
            elif row['TIC'] in existing_tic_lst:
                action = 'Added'
            else:
                action = 'Add New'
            item = (row['TIC'], row['m_TIC'], row['Disp'],
                    '{:6.2f}'.format(row['_r']),
                    '' if row['Tmag'] is np.ma.masked else '{:6.3f}'.format(row['Tmag']),
                    row['GAIA'],
                    row['Gaia3'] if row['Gaia3']>0 else '',
                    row['VarClass'],
                    '{:8.4f}'.format(row['Period']) if not np.isnan(row['Period']) else '',
                    action)
            iid = self.nearby_tree2.insert('', tk.END, values=item, tags='normal')

    def on_select_bright_tree(self, event):
        items = event.widget.selection()
        if len(items)>0:
            # clear selection in fiant cataloue
            item = self.nearby_tree2.selection()
            if item:
                self.nearby_tree2.selection_remove(item)

            item = items[0]
            self.select_bright_item(item)

    def select_bright_item(self, item):
        """event handler for clicking an item in nearby table.
        """
        tictable = self.master.master.tictable
        scatters  = self.master.master.scatters

        values = self.nearby_tree.item(item, 'values')
        tic = int(values[0])
        self.master.master.selected_tic = tic
        m = tictable['TIC']==tic
        n = len(tictable)
        linewidths = np.array([0.3]*n)
        linewidths[m] = 1
        scatters.set_linewidths(linewidths)
        self.master.canvas.draw()

    def on_select_fiant_tree(self, event):
        items = event.widget.selection()
        if len(items)>0:
            # clear selection in fiant cataloue
            item = self.nearby_tree.selection()
            if item:
                self.nearby_tree.selection_remove(item)

            item = items[0]
            self.select_fiant_item(item)

    def select_fiant_item(self, item):
        tictable = self.master.master.tictable
        scatters  = self.master.master.scatters

        values = self.nearby_tree2.item(item, 'values')
        tic = int(values[0])
        self.master.master.selected_tic = tic
        m = tictable['TIC']==tic
        n = len(tictable)
        linewidths = np.array([0.3]*n)
        linewidths[m] = 1
        scatters.set_linewidths(linewidths)
        self.master.canvas.draw()


    def select_tic(self, tic):

        found = False
        for item in self.nearby_tree.get_children():
            values = self.nearby_tree.item(item, 'value')
            if int(values[0]) == tic:
                self.nearby_tree.selection_set(item)
                found = True
                break

        if not found:
            for item in self.nearby_tree2.get_children():
                values = self.nearby_tree2.item(item, 'value')
                if int(values[0]) == tic:
                    self.nearby_tree2.selection_set(item)
                    break
            

    def on_click_neb(self, event):

        # get 
        tree = event.widget
        # tree is either self.nearby_tree(for bright) or self.nearby_tree2
        # (for dark sources)

        iid = tree.identify_row(event.y)
        column = tree.identify_column(event.x)

        if column == '#10':
            values = tree.item(iid, 'values')
            if values[9] != 'Add New':
                return
            neb_tic = int(values[0])
            gaia_class = values[7]


            # set aliases
            main_window = self.master.master
            source_frame = main_window.right_frame.source_frame
            source_tree = source_frame.source_tree

            # change the "Add New" button to "Added" in the nearby list
            values = list(values)
            values[9] = 'Added'
            tree.item(iid, values=values)
            
            # find main iid
            main_iid = source_frame.main_tic_iid_lst[main_window.tic]
            # find insert position
            item = source_tree.selection()[0]
            pos = source_tree.index(item)
            # get comment from Gaia3 Var Class
            comment = gaia_class.strip()

            # insert new item in source table
            newitem = (main_window.tic, neb_tic, '', comment, u'\xd7')
            new_iid = source_tree.insert(main_iid, pos+1,
                            values=newitem, tags='neb')
            # set background to gray and font color to gray
            source_tree.tag_configure('neb', foreground='#666666',
                                            background='#DDDDDD')
            source_tree.selection_set(new_iid)
            # set comment string into the right frame
            main_window.right_frame.comment.set(comment)

            # change status of source tree
            source_frame.file_changed = True
            # change status of save button
            source_frame.save_button['state'] = tk.NORMAL



    def get_selected_tic(self):
        # get selected TIC number
        tree1 = self.nearby_tree
        tree2 = self.nearby_tree2
        items = tree1.selection()
        if len(items)>0:
            item = items[0]
            value = tree1.item(item, 'values')
            selected_tic = int(value[0])
        else:
            items = tree2.selection()
            if len(items)>0:
                item = items[0]
                value = tree2.item(item, 'values')
                selected_tic = int(value[0])
            else:
                selected_tic = 0
        return selected_tic

            
class RightFrame(tk.Frame):
    def __init__(self, master, width, height, source_filename):
        self.master = master
        tk.Frame.__init__(self, master, width=width, height=height)


        self.source_frame = SourceTable(master = self,
                                        width  = width,
                                        height = height-300,
                                        source_filename = source_filename,
                                        )
        self.sector_frame = SectorTable(master = self,
                                        width  = width,
                                        height = 300,
                                        )

        # creat control frame
        control_frame = tk.Frame(self)

        # add text box for comments
        # create a label
        comment_label = tk.Label(
                            master = control_frame,
                            text='Comments:',
                            )
        # create a textbox
        self.comment = tk.StringVar()
        self.comment_box = tk.Entry(
                            master = control_frame,
                            textvariable = self.comment,
                            )
        self.comment.trace_add('write', self.on_add_comment)



        # set button height and width
        button_width = 90
        button_width_clear = 30
        button_height = 28

        self.aperbuttons = {}
        self.icons = {}

        icondata = {
            '2x2': '''R0lGODlhDQANAIABAAAAAP///
                yH5BAEKAAEALAAAAAANAA0AAAIbjI+pC+2OwJGG1olzsHa/F2kdp21i9EHLyh4FADs=''',
            '+': '''R0lGODlhDQANAIABAAAAAP///
                yH5BAEKAAEALAAAAAANAA0AAAIdjAOZx3jQnozOwKeyvrUvOyHciG0bQ1JptDYaUwAAOw==''',
            '3x3': '''R0lGODlhDQANAIABAAAAAP///
                yH5BAEKAAEALAAAAAANAA0AAAIdhI8YCxvN4HPS0BuTtgy/6nFTl21JSIHkh5amUQAAOw==''',
            '4x4-4': '''R0lGODlhDQANAIABAAAAAP///
                yH5BAEKAAEALAAAAAANAA0AAAIdjIEJxrF+WoTUqavm21sfDHYU543hxYhPo0oWahQAOw==''',
            '4x4': '''R0lGODlhDQANAIABAAAAAP///
                yH5BAEKAAEALAAAAAANAA0AAAIahI8YAdvsnnQxWUOhxjvf73lTx3xWuI2qiRQAOw==''',
            '5x5-4': '''R0lGODlhEAAQAIABAAAAAP///
                yH5BAEKAAEALAAAAAAQABAAAAIkjIGpxgEHXYtPWqeyrpyeboHb
                mHjhZ4rkmFJiumrMe83ttDIFADs=''',
            '6x6-12': '''R0lGODlhEwATAIABAAAAAP///
                yH5BAEKAAEALAAAAAATABMAAAIyjA+pcO0cIHTPSIqWVvRGC1bf
                6EXbuYXeuqoqCnOkS7ZzHaMIXe53b8lNOjSMD4g5UQoAOw==''',
                }

        self.icons['2x2'] = tk.PhotoImage(data = icondata['2x2'])
        self.aperbuttons['2x2'] = tk.Button(
                        master  = control_frame,
                        text    = '',
                        width   = button_width,
                        height  = button_height,
                        image   = self.icons['2x2'],
                        command = lambda: self.set_aperture('2x2'),
                        state   = tk.DISABLED,
                        )
        self.icons['+'] = tk.PhotoImage(data = icondata['+'])
        self.aperbuttons['+'] = tk.Button(
                        master  = control_frame,
                        text    = '',
                        width   = button_width,
                        height  = button_height,
                        image   = self.icons['+'],
                        command = lambda: self.set_aperture('+'),
                        state   = tk.DISABLED,
                        )
        self.icons['3x3'] = tk.PhotoImage(data = icondata['3x3'])
        self.aperbuttons['3x3'] = tk.Button(
                        master  = control_frame,
                        text    = '',
                        width   = button_width,
                        height  = button_height,
                        image   = self.icons['3x3'],
                        command = lambda: self.set_aperture('3x3'),
                        state   = tk.DISABLED,
                        )
        self.icons['4x4-4'] = tk.PhotoImage(data = icondata['4x4-4'])
        self.aperbuttons['4x4-4'] = tk.Button(
                        master  = control_frame,
                        text    = '',
                        width   = button_width,
                        height  = button_height,
                        image   = self.icons['4x4-4'],
                        command = lambda: self.set_aperture('4x4-4'),
                        state   = tk.DISABLED,
                        )
        self.icons['4x4'] = tk.PhotoImage(data = icondata['4x4'])
        self.aperbuttons['4x4'] = tk.Button(
                        master  = control_frame,
                        text    = '',
                        width   = button_width,
                        height  = button_height,
                        image   = self.icons['4x4'],
                        command = lambda: self.set_aperture('4x4'),
                        state   = tk.DISABLED,
                        )

        self.icons['5x5-4'] = tk.PhotoImage(data = icondata['5x5-4'])
        self.aperbuttons['5x5-4'] = tk.Button(
                        master  = control_frame,
                        text    = '',
                        width   = button_width,
                        height  = button_height,
                        image   = self.icons['5x5-4'],
                        command = lambda: self.set_aperture('5x5-4'),
                        state   = tk.DISABLED,
                        )
        self.icons['6x6-12'] = tk.PhotoImage(data = icondata['6x6-12'])
        self.aperbuttons['6x6-12'] = tk.Button(
                        master  = control_frame,
                        text    = '',
                        width   = button_width,
                        height  = button_height,
                        image   = self.icons['6x6-12'],
                        command = lambda: self.set_aperture('6x6-12'),
                        state   = tk.DISABLED,
                        )

        # clear aperture button
        self.clearap_button = tk.Button(master  = control_frame,
                                        text    = 'Clear Aperture',
                                        width   = button_width_clear,
                                        height  = 1,
                                        command = self.clear_aperture,
                                        state   = tk.DISABLED,
                                        )

        # create custom light curve save button
        self.save_lc_button = tk.Button(master = control_frame,
                                        text = 'Save New Lightcurve',
                                        #width = 150,
                                        command = self.save_new_lc,
                                        state   = tk.DISABLED,
                                        )

        # pack comment label and Entry
        comment_label.grid(row=0, column=0, sticky=tk.EW)
        self.comment_box.grid(row=0, column=1, columnspan=4, sticky=tk.EW,
                padx=1, pady=5)

        # pack the first row
        for i, key in enumerate(['2x2', '+', '3x3', '4x4-4', '4x4']):
            self.aperbuttons[key].grid(row=1, column=i, sticky=tk.EW)
        # pack the second row
        for i, key in enumerate(['5x5-4', '6x6-12']):
            self.aperbuttons[key].grid(row=2, column=i, sticky=tk.EW)
        # pack the clear button
        self.clearap_button.grid(row=2, column=2, columnspan=3, sticky=tk.EW)
        #seph1 = ttk.Separator(master=control_frame, orient='horizontal')
        #seph1.grid(row=3, column=0, columnspan=5, sticky=tk.EW, pady=5)
        # pack the savelc  button
        self.save_lc_button.grid(row=3, column=0, columnspan=5, sticky=tk.EW,
                    padx=1, pady=5)

        # pack all widgets in the right frame
        self.source_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.sector_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        control_frame.pack(side=tk.TOP, pady=2)

        self.pack()

    def on_add_comment(self, *args):

        # set source table alias
        source_tree = self.source_frame.source_tree
        # get current selected item in source tree
        item = source_tree.selection()
        # get values in source tree
        values = list(source_tree.item(item, 'values'))

        # set new comments
        values[3] = self.comment.get()
        source_tree.item(item, values=values)

        # change status of source tree
        self.source_frame.file_changed = True
        # change status of save button
        self.source_frame.save_button['state'] = tk.NORMAL

    def save_new_lc(self):
     
        ## get selected TIC number
        selected_tic = self.master.plot_frame.nearby_frame.get_selected_tic()

        # prepare filename
        fname = 'tess-{:012d}-{:012d}-s{:03d}-s_lc.fits'.format(
                selected_tic,
                self.master.tic,
                self.master.sector,
                )
        path = os.path.join(datapool, 'sa-lc', 's{:03d}'.format(
                self.master.sector))
        if not os.path.exists(path):
            os.makedirs(path)

        filename = os.path.join(path, fname)

        # prepare alias
        t_lst     = self.master.lcdata.t_lst
        tcorr_lst = self.master.lcdata.tcorr_lst
        flux_lst  = self.master.custom_flux
        bkg_lst   = np.zeros_like(flux_lst, dtype=np.float32)
        q_lst     = self.master.lcdata.q_lst
        imagemask = self.master.lcdata.imagemask
        custom_apertures = self.master.custom_apertures

        # pack light curve table
        lc_table = Table()
        lc_table.add_column(t_lst, name = 'TIME')
        lc_table.add_column(tcorr_lst, name = 'TIMECORR')
        lc_table.add_column(flux_lst, name = 'SAP_FLUX')
        lc_table.add_column(bkg_lst, name = 'SAP_BKG')
        lc_table.add_column(q_lst, name='QUALITY')


        # pack aperutere mask
        newmask = np.zeros_like(custom_apertures, dtype=np.int32)
        newmask += np.int32(imagemask & 1 > 0)*1        # pixels collected
        newmask += np.int32(custom_apertures)*2         # aperture
        newmask += np.int32(imagemask & 32 > 0)*32      # CCD output A
        newmask += np.int32(imagemask & 64 > 0)*64      # CCD output B
        newmask += np.int32(imagemask & 128 > 0)*128    # CCD output C
        newmask += np.int32(imagemask & 256 > 0)*256    # CCD output D

        # prepare HDU List
        hdulst = fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.BinTableHDU(data=lc_table),
                    fits.ImageHDU(data=newmask),
                    ])
        hdulst.writeto(filename, overwrite=True)

        print('Shifted aperture LC written to',filename)

        #### save figure
        figname = 'fig-sa-lc-{:012d}-{:012d}-s{:03d}.png'.format(
                selected_tic,
                self.master.tic,
                self.master.sector,
                )
        path = os.path.join(datapool, 'sa-fig', '{:02d}'.format(selected_tic%100))
        if not os.path.exists(path):
            os.makedirs(path)
        figfilename = os.path.join(path, figname)

        self.master.plot_frame.fig.savefig(figfilename)
        print('Fig saved to', figfilename)


        # update source table
        # set alias
        source_tree = self.source_frame.source_tree

        for item in source_tree.get_children():
            values = source_tree.item(item, 'value')
            if int(values[0])==self.master.tic:
                for item2 in source_tree.get_children(item):
                    values = list(source_tree.item(item2, 'value'))
                    if len(values[1])>0 and int(values[1])==selected_tic:
                        # get sectors
                        sectors = find_saved_sa_files(self.master.tic, selected_tic)
                        sector_string = sector_lst_to_string(sectors)
                        # update the sector list in source table
                        values[2] = sector_string
                        source_tree.item(item2, values=values)
                break


        # change status of source tree
        self.source_frame.file_changed = True
        # change status of save button
        self.source_frame.save_button['state'] = tk.NORMAL

    def clear_aperture(self):
        self.master.reset_custom_lc()
        # re-extract light curves
        self.master.extract_lc()
        self.master.plot_frame.canvas.draw()

    def set_aperture(self, shape):
        # set aliases
        tictable = self.master.tictable
        selected_tic = self.master.selected_tic
        wcoord = self.master.wcoord1
        ny, nx = self.master.custom_apertures.shape

        m = tictable['TIC']==selected_tic
        ticrow = tictable[m][0]
        ra = ticrow['RAJ2000']
        dec = ticrow['DEJ2000']
        x, y = wcoord.all_world2pix(ra, dec, 0)
        self.master.custom_apertures = np.zeros((ny, nx), dtype=bool)

        if shape in ['+', '3x3', '5x5-4']:
            ix = int(np.round(x))
            iy = int(np.round(y))

            if shape=='+':
                pixel_lst = [
                                       (ix, iy-1),
                            (ix-1,iy), (ix, iy),   (ix+1,iy),
                                       (ix, iy+1),
                                     ]
            elif shape=='3x3':
                pixel_lst = [
                    (ix-1, iy-1), (ix, iy-1), (ix+1, iy-1),
                    (ix-1, iy),   (ix, iy),   (ix+1, iy),
                    (ix-1, iy+1), (ix, iy+1), (ix+1, iy+1),
                    ]
            elif shape=='5x5-4':
                pixel_lst = [
                                  (ix-1, iy-2), (ix, iy-2), (ix+1, iy-2),
                    (ix-2, iy-1), (ix-1, iy-1), (ix, iy-1), (ix+1, iy-1), (ix+2, iy-1),
                    (ix-2, iy),   (ix-1, iy),   (ix, iy),   (ix+1, iy),   (ix+2, iy),
                    (ix-2, iy+1), (ix-1, iy+1), (ix, iy+1), (ix+1, iy+1), (ix+2, iy+1),
                                  (ix-1, iy+2), (ix, iy+2), (ix+1, iy+2),
                    ]
            else:
                pixel_lst = []

            for _ix, _iy in pixel_lst:
                if 0<=_ix<nx and 0<=_iy<ny:
                    self.master.custom_apertures[_iy, _ix] = True


        elif shape in ['2x2', '4x4', '4x4-4', '6x6-12']:
            ix = int(np.round(x))
            iy = int(np.round(y))
            if ix > x:
                ix = ix - 0.5
            else:
                ix = ix + 0.5
            if iy > y:
                iy = iy - 0.5
            else:
                iy = iy + 0.5

            if shape=='2x2':
                pixel_lst = [
                    (ix-0.5, iy-0.5), (ix+0.5, iy-0.5),
                    (ix-0.5, iy+0.5), (ix+0.5, iy+0.5),
                    ]
            elif shape=='4x4-4':
                pixel_lst = [
                                   (ix-0.5,iy-1.5),(ix+0.5,iy-1.5),
                   (ix-1.5,iy-0.5),(ix-0.5,iy-0.5),(ix+0.5,iy-0.5),(ix+1.5,iy-0.5),
                   (ix-1.5,iy+0.5),(ix-0.5,iy+0.5),(ix+0.5,iy+0.5),(ix+1.5,iy+0.5),
                                   (ix-0.5,iy+1.5),(ix+0.5,iy+1.5),
                   ]

            elif shape=='4x4':
                pixel_lst = [
                   (ix-1.5,iy-1.5),(ix-0.5,iy-1.5),(ix+0.5,iy-1.5),(ix+1.5,iy-1.5),
                   (ix-1.5,iy-0.5),(ix-0.5,iy-0.5),(ix+0.5,iy-0.5),(ix+1.5,iy-0.5),
                   (ix-1.5,iy+0.5),(ix-0.5,iy+0.5),(ix+0.5,iy+0.5),(ix+1.5,iy+0.5),
                   (ix-1.5,iy+1.5),(ix-0.5,iy+1.5),(ix+0.5,iy+1.5),(ix+1.5,iy+1.5),
                   ]
            elif shape=='6x6-12':
                pixel_lst = [
                                                   (ix-0.5,iy-2.5),(ix+0.5,iy-2.5),
                                   (ix-1.5,iy-1.5),(ix-0.5,iy-1.5),(ix+0.5,iy-1.5),(ix+1.5,iy-1.5),
                   (ix-2.5,iy-0.5),(ix-1.5,iy-0.5),(ix-0.5,iy-0.5),(ix+0.5,iy-0.5),(ix+1.5,iy-0.5),(ix+2.5,iy-0.5),
                   (ix-2.5,iy+0.5),(ix-1.5,iy+0.5),(ix-0.5,iy+0.5),(ix+0.5,iy+0.5),(ix+1.5,iy+0.5),(ix+2.5,iy+0.5),
                                   (ix-1.5,iy+1.5),(ix-0.5,iy+1.5),(ix+0.5,iy+1.5),(ix+1.5,iy+1.5),
                                                   (ix-0.5,iy+2.5),(ix+0.5,iy+2.5),
                   ]
            else:
                pixel_lst = []

            for _ix, _iy in pixel_lst:
                _ix = int(_ix)
                _iy = int(_iy)
                if 0<=_ix<nx and 0<=_iy<ny:
                    self.master.custom_apertures[_iy, _ix] = True


        self.master.custom_img.set(data=self.master.custom_apertures)

        # set aperture plot in the center PBP axes
        for (y, x), ax in self.master.pbp_axes.items():
            if self.master.custom_apertures[y, x]:
                ax.patch.set_facecolor('#ffbbdd')
            else:
                ax.patch.set_facecolor('w')
        
        self.master.extract_lc()
        self.master.plot_frame.canvas.draw()
        

    def set_aperture_button(self, state):
        if state:
            for shape, button in self.aperbuttons.items():
                button['state'] = tk.NORMAL
        else:
            for shape, button in self.aperbuttons.items():
                button['state'] = tk.DISABLED


class SourceTable(tk.Frame):
    def __init__(self, master, width, height, source_filename):
        self.master = master
        self.source_filename = source_filename

        self.source_table = Table.read(source_filename,
                            format='ascii.fixed_width_two_line')

        # add a column named "real_TIC"
        if 'real_TIC' not in self.source_table.colnames:
            n = len(self.source_table)
            pos = self.source_table.colnames.index('comments')
            self.source_table.add_column([np.ma.masked]*n,
                    name='real_TIC', index=pos)

        # get sector lst for each main TIC
        # pick up rows with main TICs
        m = np.array([v is np.ma.masked for v in
                            list(self.source_table['real_TIC'])])
        tic_lst = list(self.source_table[m]['TIC'])
        sector_lst = get_lc_sectors(tic_lst)

        self.file_changed = False

        tk.Frame.__init__(self, master, width=width, height=height)
        
        self.source_tree = ttk.Treeview(
                master  = self,
                columns = ('TIC', 'real_TIC', 'sectors', 'note', 'action'),
                show    = 'tree headings',
                style   = 'Treeview',
                height  = 20,
                selectmode = 'browse',
                )
        self.source_tree.bind('<<TreeviewSelect>>', self.on_select_item)
        self.source_tree.bind('<Button-1>', self.on_click)


        self.scrollbar = tk.Scrollbar(master=self,
                orient = tk.VERTICAL,
                width  = 10,
                )
        self.source_tree.column('#0', width=20, anchor='center', stretch=False)
        self.source_tree.column('TIC',      width=90, anchor='e')
        #self.source_tree.column('m_TIC',    width=90, anchor='e')
        #self.source_tree.column('Disp',     width=80)
        self.source_tree.column('real_TIC', width=90, anchor='e')
        self.source_tree.column('sectors',  width=160)
        self.source_tree.column('note',     width=width-400)
        self.source_tree.column('action',   width=20)

        self.source_tree.heading('#0', text='')
        self.source_tree.heading('TIC',text='TIC')
        #self.source_tree.heading('m_TIC',text='m_TIC')
        #self.source_tree.heading('Disp',text='Disp')
        self.source_tree.heading('real_TIC',text='real_TIC')
        self.source_tree.heading('sectors', text='sectors')
        self.source_tree.heading('note', text='note')
        self.source_tree.heading('action', text='')

        self.source_tree.config(yscrollcommand = self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=30)

        self.scrollbar.config(command=self.source_tree.yview)

        self.main_tic_iid_lst = {}
        for row in self.source_table:

            tic = row['TIC']
            real_tic = row['real_TIC']
            if row['comments'] is np.ma.masked:
                comments = ''
            else:
                comments = row['comments']

            if real_tic is np.ma.masked:
                # this is a main TIC
                sectors = sector_lst[tic]
                sector_string = sector_lst_to_string(sectors)
                item = (tic, '', sector_string, comments, '')
                iid = self.source_tree.insert('', tk.END,
                        values=item, tags='normal', open=True)
                self.main_tic_iid_lst[tic] = iid
            else:
                # this is not a main tic
                sectors = find_saved_sa_files(tic, real_tic)
                sector_string = sector_lst_to_string(sectors)
                item = (tic, real_tic, sector_string, comments, u'\xd7')
                main_iid = self.main_tic_iid_lst[tic]
                iid = self.source_tree.insert(main_iid, tk.END,
                        values=item, tags='neb', open=True)

        # set background to gray and font color to gray
        self.source_tree.tag_configure('neb', foreground='#666666',
                                              background='#DDDDDD')
            

        # add a save button
        self.save_button = tk.Button(master = self,
                                     text = 'Save List',
                                     width = 150,
                                     command = self.save_source_table,
                                     state  = tk.DISABLED,
                                     )

        self.save_button.pack(side=tk.TOP)
        self.source_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # add righ-clicked menu
        #self.source_tree.bind('<Button-3>', self.on_pop_menu)

        self.pack()


    def on_select_item(self, event):
        """Event handler for clicking an item in source table.
        """

        items = event.widget.selection()
        if len(items)==0:
            return None

        item = items[0]
        values = self.source_tree.item(item, 'values')
        tic = int(values[0])
        comment = values[3]
        
        if len(values[1])==0:
            # this tic is a main TIC
            selected_tic = tic
        else:
            # this tic is not a main TIC
            tic = int(values[0])
            selected_tic = int(values[1])

        if tic == self.master.master.tic:
            # same main TIC
            # no need to refresh the displayed TP and PBP
            if selected_tic == tic:
                self.master.master.reset_custom_lc()
            else:
                self.master.master.plot_selected_tic(tic,
                        self.master.master.sector, selected_tic)

            # will triger plot function
            self.master.master.plot_frame.nearby_frame.select_tic(selected_tic)

        else:
            # when getting new stars, reset set_nearby
            self.master.master.set_nearby = False
            # reset sector list
            sectors = get_lc_sectors(tic)
            # load sector list in sector tree
            # this will trigger on_select_item function, the will trigger:
            # load_nearby_table, and plot function
            self.master.sector_frame.load_sectors(sectors)


        # set status of aperture buttons
        self.master.set_aperture_button(True)

        # put comments into comment box
        self.master.comment.set(comment)

    def on_click(self, event):
        source_tree = event.widget
        iid = source_tree.identify_row(event.y)
        column = source_tree.identify_column(event.x)

        if column == '#5':
            # get parent of this item
            main_iid = self.source_tree.parent(iid)

            values = self.source_tree.item(iid, 'values')

            tic = int(values[0])
            selected_tic = int(values[1])
            sector_string = values[2]

            sector_lst = sector_string_to_lst(sector_string)
            for sector in sector_lst:
                # delete lc file
                fname = 'tess-{:012d}-{:012d}-s{:03d}-s_lc.fits'.format(
                            selected_tic, tic, sector)
                filename = os.path.join(datapool, 'sa-lc',
                            's{:03d}'.format(sector), fname)
                if os.path.exists(filename):
                    print('Delete', filename)
                    os.remove(filename)
                # delete figure
                figname = 'fig-sa-lc-{:012d}-{:012d}-s{:03d}.png'.format(
                            selected_tic, tic, sector)
                figfilename = os.path.join(datapool, 'sa-fig',
                            '{:02d}'.format(selected_tic%100), figname)
                if os.path.exists(figfilename):
                    print('Delete', figfilename)
                    os.remove(figfilename)

            # delete item from source tree
            self.source_tree.delete(iid)
            # select the parent item
            #self.source_tree.selection_set(main_iid)

            # change status of source tree
            self.file_changed = True
            # change status of save button
            self.save_button['state'] = tk.NORMAL
    

    def on_pop_menu(self, event):
        """Pop up a menu on source table"""

        iid = self.source_tree.identify_row(event.y)
        self.source_tree.selection_set(iid)
        source_menu = tk.Menu(self, tearoff=0)
        source_menu.add_command(label='Delete', command=self.delete_item)
        #source_menu.post(event.x_root, event.y_root)
        try:
            source_menu.tk_popup(event.x_root, event.y_root)
        finally:
            source_menu.grab_release()
            

    def delete_item(self):

        item = self.source_tree.selection()[0]
        self.source_tree.delete(item)

        # change status of source tree
        self.file_changed = True
        # change status of save button
        self.save_button['state'] = tk.NORMAL

            
    def save_source_table(self):


        tic_lst, real_tic_lst, comment_lst = [], [], []
        for item in self.source_tree.get_children():
            values = self.source_tree.item(item, 'value')

            tic_lst.append(int(values[0]))
            real_tic_lst.append(np.ma.masked)
            comment_lst.append(values[3].strip())
            

            for item2 in self.source_tree.get_children(item):
                values = self.source_tree.item(item2, 'value')

                tic_lst.append(int(values[0]))
                real_tic_lst.append(int(values[1]))
                comment_lst.append(values[3].strip())

        newtable = Table()
        newtable.add_column(tic_lst, name='TIC')
        newtable.add_column(real_tic_lst, name='real_TIC')
        newtable.add_column(comment_lst, name='comments')

        maxlen = max([len(v) for v in newtable['comments']
                        if v is not np.ma.masked])
        newtable['comments'].info.format='%-{}s'.format(maxlen)

        newtable.write(self.source_filename, 
                format='ascii.fixed_width_two_line', overwrite=True)
        print('Source Table written to {}'.format(self.source_filename))

        # reset status
        self.file_changed = False
        self.save_button['state'] = tk.DISABLED


def launch(source_filename, datapool_path):

    global datapool
    datapool = datapool_path

    master = tk.Tk()
    master.resizable(width=False, height=False)

    screen_width  = master.winfo_screenwidth()
    screen_height = master.winfo_screenheight()

    if sys.platform.startswith('linux'):
        window_width = int(screen_width-200)
        window_height = int(screen_height-200)
    elif sys.platform.startswith('darwin'):
        window_width = int(screen_width-2000)
        window_height = int(screen_height-1000)
    else:
        print('Unrecognized OS:', sys.platform)
        exit()

    if window_width > window_height*3:
        window_width = window_height*3
    else:
        window_height = window_width//3

    x = int((screen_width-window_width)/2.)
    y = int((screen_height-window_height)/2.)

    master.geometry('{}x{}+{}+{}'.format(window_width, window_height, x, y))


    mainwindow = MainWindow(master,
                            width  = window_width,
                            height = window_height,
                            source_filename = source_filename,
                            )

    master.mainloop()


def get_mixed_table(tic, tictable,
        #gaia2table, gaia3table,
        gaia3vartable):
    
    columns = ['TIC', 'm_TIC', 'Disp', '_r', 'Tmag', 'GAIA']
    mixed_table = tictable[columns]

    #coord3_lst = SkyCoord(gaia3table['RA_ICRS'], gaia3table['DE_ICRS'],
    #                unit='deg')
    #
    #mask_gaia3 = [False]*len(gaia3table)
    #ra_lst, dec_lst = [], []
    #pmra_lst, pmdec_lst = [], []
    #gaia3_lst = []
    #plx_lst = []
    #for row in mixed_table:
    #    if row['GAIA'] is np.ma.masked:
    #        gaia3_lst.append(0)
    #        plx_lst.append(0.0)
    #        continue
    #    m = gaia2table['Source']==row['GAIA']
    #    if m.sum()==0:
    #        gaia3_lst.append(0)
    #        plx_lst.append(0.0)
    #        continue
    #    row2 = gaia2table[m][0]
    #    ra  = row2['RA_ICRS']
    #    dec = row2['DE_ICRS']
    #    pmra = row2['pmRA']
    #    if pmra is np.ma.masked:
    #        pmra = 0.0
    #    pmdec = row2['pmDE']
    #    if pmdec is np.ma.masked:
    #        pmdec = 0.0
    #    dt = 0.5
    #    newra = ra + pmra*1e-3/3600/np.cos(np.deg2rad(dec))*dt
    #    newdec = dec + pmdec*1e-3/3600*dt

     #   coord = SkyCoord(newra, newdec, unit='deg')
     #   sep = coord.separation(coord3_lst)
     #   m3 = sep.arcsec < 0.1
     #   if m3.sum()==1:
     #       row3 = gaia3table[m3][0]
     #       gaia3 = row3['Source']
     #       plx   = row3['Plx']
     #   else:
     #       gaia3 = 0
     #       plx = 0.0
     #   gaia3_lst.append(gaia3)
     #   plx_lst.append(plx)
    #mixed_table.rename_column('GAIA', 'Gaia2')
    #mixed_table.add_column(gaia3_lst, name='Gaia3')
    #mixed_table.add_column(plx_lst, name='Plx')

    #for row3v in gaia3vartable:
    #    print(row3v['RAJ2000'], row3v['DEJ2000'])
    coord_lst = SkyCoord(tictable['RAJ2000'], tictable['DEJ2000'], unit='deg')
    n = len(tictable)
    gaia3_lst = np.array([0]*n)
    vartype_lst = np.array([' '*20]*n)
    period_lst = np.array([np.nan]*n)
    for row3v in gaia3vartable:
        coord = SkyCoord(row3v['RA_ICRS'], row3v['DE_ICRS'], unit='deg')
        sep = coord_lst.separation(coord)
        mini = sep.arcsec.argmin()
        if sep.arcsec[mini] < 3.0:
            gaia3_lst[mini] = row3v['Source']
            vartype_lst[mini] = row3v['Class']
            if row3v['Class']=='ECL':
                catid = 'I/358/veb'
                column_filter = {'Source':'={}'.format(row3v['Source'])}
                viz = Vizier(catalog=catid, columns=['**'],
                        column_filters=column_filter)
                tablelist = viz.query_constraints()
                tab = tablelist[catid]
                if tab is not None and len(tab)>0:
                    _row = tab[0]
                    period_lst[mini] = 1/_row['Freq']
    mixed_table.add_column(gaia3_lst, name='Gaia3')
    mixed_table.add_column(vartype_lst, name='VarClass')
    mixed_table.add_column(period_lst, name='Period')
                
    m1 = tictable['TIC']==tic
    m2 = [v is not np.ma.masked and v==tic for v in tictable['m_TIC']]
    # set magnitude criteria
    #m3 = tictable['Tmag'] < 18
    m3 = tictable['Tmag'] < 17
    m = m1 + m2 + m3
    return mixed_table[m], mixed_table[~m]

