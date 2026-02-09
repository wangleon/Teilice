import re
import os
import io
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
from astropy.table import Table, MaskedColumn
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
from ..periodogram import GLS


def consecutive(data):
    return np.split(data, np.where(np.diff(data) != 1)[0]+1)

def get_lc(tic, sector_lst, datapool):
    lc_lst = {}
    for sector in sector_lst:
        path = os.path.join(datapool, 'lc', 's{:03d}'.format(sector))

        # get lc file
        find = False
        for fname in os.listdir(path):
            _tic = int(fname.split('-')[2])
            if _tic == tic:
                find = True
                break
        if not find:
            print('Error: TIC {} file does not exist in Sector {}'.format(tic, sector))
            raise ValueError

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

def get_cont_lst(t_lst, tol):
    cont_lst = []
    m = np.diff(t_lst) < tol
    ii = np.nonzero(np.int16(m))[0]
    for group in np.split(ii, np.where(np.diff(ii)>1)[0]+1):
        i1, i2 = group[0], group[-1]
        t1 = t_lst[i1]
        t2 = t_lst[i2]
        cont_lst.append((t1, t2))
    return cont_lst



class MainWindow(tk.Frame):

    def __init__(self, master, width, height, source_filename, datapool,
            figure_path, folded_path):

        self.master = master
        self.datapool = datapool
        self.figure_path = figure_path
        self.folded_path = folded_path

        tk.Frame.__init__(self, master, width=width, height=height)

        dpi = 200

        right_width = 500


        # flags and their explanations
        self.flag_lst = [
                ('oce', "O'Connell Effect"),
                ('meb', 'Multiple Eclipses'),
                ('etv', 'Eclipse Timing Variation'),
                ('oeb', 'Oscillating EB'),
                ('sgl', 'Single Eclipse'),
                #('flr', 'Flare'),
                ]

        self.plot_frame = PlotFrame(master=self,
                                    width = width - right_width,
                                    height = height - 300,
                                    dpi = dpi,
                                    )
        self.right_frame = RightFrame(master=self,
                                      width = right_width,
                                      height = height,
                                      source_filename = source_filename,
                                      )
        self.plot_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.pack()

        self.reset_param()
        self.plot_frame.control_panel.reset_param()


    def reset_param(self):

        # initialize parameters
        self.tic = -1

        self.display_urej = 100 # upper rejction for display Y axis
        self.correct_offset = True # if True, correct sector-by-sector offset
        # reset control buttons
        #self.plot_frame.control_panel.urej.set(self.display_urej)
        #self.plot_frame.control_panel.correct_offset.set(self.correct_offset)

        self.sector_lst = [] # sector list
        self.sector_mask = [] # mask for displaying sectors
        self.lc_lst = {}
        # median flux of each sector
        self.medflux_lst = {}
        # tspan_lst is a dict containing the (t1, t2) of each sector
        self.tspan_lst = {}
        # lc mask is dict containing the masked time of each sector
        self.lc_mask = {}
        # tlength_lst is a dict containing the unblocked time length of each
        # sector. And it should be updated after self.lc_mask changed
        self.tlength_lst = {}
        # lc trend
        self.lc_trends = None
        self.detrendwin = 20
        # modeled light curve
        self.model_curve = {'primary': None, 'secondary': None}

        # segment info
        self.segment_lst = []
        self.seglc_lst = {}
        self.all_lc = () # all_lc combined together
        self.medflux = 0.0
        self.offset_lst = {}
        self.argseg_lst = {}
        # GLS
        ngls = 500
        self.gls_period_lst = np.logspace(-1, 2, ngls)
        self.gls_lst = {}
        self.all_gls = np.zeros(ngls, dtype=np.float32)

        # orbital parameters
        self.period = np.nan # orbital period
        self.t0 = np.nan     # t0
        self.secphase = 0.5  # phase of secondary eclipse
        self.priwin = 0.05 # window of primary eclipse
        self.secwin = 0.05 # window of secondary eclipse


        self.source_changed = False


    def read_lc(self):

        self.sector_lst = get_lc_sectors(self.tic)
        self.sector_mask = {s: True for s in self.sector_lst}
        self.lc_lst = get_lc(self.tic, self.sector_lst, self.datapool)

        # initialize tspan_lst
        self.tspan_lst = {s: (t_lst[0], t_lst[-1])
                        for s, (t_lst, f_lst) in self.lc_lst.items()}

        # initialize lc_mask
        for sector in self.sector_lst:
            self.lc_mask[sector] = []
        self.find_tlength()  # this will update the self.tlength_lst

        # find median flux of each sector
        self.medflux_lst = {s: np.median(f_lst)
                            for s, (t_lst, f_lst) in self.lc_lst.items()
                            }

        # generate segments
        self.segment_lst = consecutive(self.sector_lst)

        # determine each segment belong to which segment
        for iseg, segs in enumerate(self.segment_lst):
            for sector in segs:
                self.argseg_lst[sector] = iseg

    def get_effective_lc(self, sector):
        """get effective t_lst of a give sector"""
        dataitem = self.lc_lst[sector]
        t_lst, f_lst = dataitem
        mask = np.ones_like(t_lst, dtype=bool)
        # True means unblocked. False means blocked

        for t1, t2 in self.lc_mask[sector]:
            m1 = t_lst >= t1
            m2 = t_lst <= t2
            m = m1 * m2
            mask[m] = False
        return t_lst[mask], f_lst[mask]

    def find_tlength(self):
        # find total time of span
        tol = 5             # tolerance of cadence ratio
        cadence = 2/60/24   # translate cadence from minute to day,
        for sector, dataitem in self.lc_lst.items():
            # use unblocked light curve
            eff_t_lst, eff_f_lst = self.get_effective_lc(sector)
            # get continous list
            cont_lst = get_cont_lst(eff_t_lst, tol*cadence)
            # count total length
            tlength_total = sum([t2-t1 for t1, t2 in cont_lst])
            self.tlength_lst[sector] = tlength_total

    def find_offset(self):

        self.medflux = np.median(list(self.medflux_lst.values()))

        self.offset_lst = {sector: v - self.medflux
                            for sector, v in self.medflux_lst.items()
                            }

    def count_neclipse(self):
        sector_used = [s for s in self.sector_lst if self.sector_mask[s]]

        npri, nsec = 0, 0
        t1_pri, t2_pri = 0.0, 0.0
        t1_sec, t2_sec = 0.0, 0.0

        for sector in sector_used:
            eff_t_lst, eff_f_lst = self.get_effective_lc(sector)
            # get continuous list
            tol_pri = self.priwin * self.period / 2
            tol_sec = self.secwin * self.period / 2
            tol = min(tol_pri, tol_sec)

            cont_lst = get_cont_lst(eff_t_lst, tol)
            t1 = eff_t_lst[0]
            t2 = eff_t_lst[-1]
            i1 = int((t1 - self.t0)/self.period)
            i2 = int((t2 - self.t0)/self.period)
            for i in np.arange(i1-1, i2+1):
                _tpri = self.t0 + i * self.period
                _tsec = self.t0 + (i + self.secphase) * self.period
                for _t1, _t2 in cont_lst:
                    if _t1 < _tpri < _t2:
                        if npri == 0:
                            t1_pri = _tpri
                        npri += 1
                        # refresh the time of first primary eclipse
                        t2_pri = _tpri
                    if _t1 < _tsec < _t2:
                        if nsec == 0:
                            t1_sec = _tsec
                        nsec += 1
                        # refresh the time of last secondary eclipse
                        t2_sec = _tsec
                    if _t1 > _tpri and _t2 > _tsec:
                        break

        tspan_pri = t2_pri - t1_pri
        tspan_sec = t2_sec - t1_sec
        return npri, nsec, tspan_pri, tspan_sec

    def get_seg_lc(self):

        for segs in self.segment_lst:
            if len(segs)==1:
                # if there is only one sector in this segment
                s1 = segs[0]
                key = (s1, s1)
                t_lst, f_lst = self.lc_lst[s1]
                foffset = self.offset_lst[s1]
                self.seglc_lst[key] = (t_lst, f_lst - foffset)
            else:
                s1, s2 = segs[0], segs[-1]
                key = (s1, s2)
                # make the lightcurve data for the entire segment
                tmpt, tmpf = [], []
                for sector in segs:
                    t_lst, f_lst = self.lc_lst[sector]
                    foffset = self.offset_lst[sector]
                    # append the lc data to the lc data for the entire segment
                    tmpt.append(t_lst)
                    tmpf.append(f_lst - foffset)
                self.seglc_lst[(s1, s2)] = (
                        np.concatenate(tmpt),
                        np.concatenate(tmpf)
                        )
        # get all lc combined together
        tmpt, tmpf = [], []

        for segs in self.segment_lst:
            s1, s2 = segs[0], segs[-1]
            t_lst, f_lst = self.seglc_lst[(s1, s2)]
            tmpt.append(t_lst)
            tmpf.append(f_lst)
        self.all_lc = (
                np.concatenate(tmpt),
                np.concatenate(tmpf),
                )


    def calc_gls(self):

        for (s1, s2), dataitem in self.seglc_lst.items():
            nsector = s2 - s1 + 1
            seg_t_lst, seg_f_lst = dataitem
            m = seg_f_lst < np.percentile(seg_f_lst, 99)
            pdm = GLS(seg_t_lst[m], seg_f_lst[m])
            power, _ = pdm.get_power(period=self.gls_period_lst)
            self.gls_lst[(s1, s2)] = power

            self.all_gls += power * nsector


    def find_period_auto(self):
        imax = self.all_gls.argmax()
        maxperiod = self.gls_period_lst[imax]
        self.period = maxperiod

    def find_t0_auto(self):
        # find T0
        all_t_lst, all_f_lst = self.all_lc
        imin = all_f_lst.argmin()
        tmin = all_t_lst[imin]  # time of minimum flux
        #self.t0 = tmin - int((tmin - all_t_lst[0])/self.period)*self.period
        self.t0 = tmin

    def change_period(self, ratio):
        self.set_period(self.period * ratio)
        # refresh period
        #self.plot_frame.control_panel.update_param()
        #self.plot()
        #self.plot_frame.canvas.draw()

    def set_period(self, p):
        self.period = p
        # refresh period
        self.plot_frame.control_panel.update_param()
        self.plot()
        self.plot_frame.canvas.draw()

    def change_t0(self, level):
        self.t0 = self.t0 + self.period * level
        # refresh t0
        self.plot_frame.control_panel.update_param()
        self.plot()
        self.plot_frame.canvas.draw()

    def set_t0(self, t0):
        self.t0 = t0
        # refresh t0
        self.plot_frame.control_panel.update_param()
        self.plot()
        self.plot_frame.canvas.draw()

    def set_secphase(self, secphase):
        self.secphase = secphase
        # refresh secphase
        print(self.secphase)
        self.plot_frame.control_panel.update_param()
        self.plot()
        self.plot_frame.canvas.draw()

    def change_priwin(self, ratio):
        self.priwin = self.priwin * ratio
        self.plot()
        self.plot_frame.canvas.draw()

    def change_secwin(self, ratio):
        self.secwin = self.secwin * ratio
        self.plot()
        self.plot_frame.canvas.draw()

    def change_secphase(self, v):
        self.secphase = self.secphase + v
        self.plot_frame.control_panel.update_param()
        self.plot()
        self.plot_frame.canvas.draw()

    def change_urej(self, urej):
        self.display_urej = urej
        self.plot()
        self.plot_frame.canvas.draw()

    def change_offset(self, v):
        self.correct_offset = v
        self.plot()
        self.plot_frame.canvas.draw()


    def change_detrendwin(self, v):
        self.detrendwin = v
        self.fit_ooe()

    def fit_ooe(self):

        self.lc_trends = {}

        if ~np.isnan(self.period) and ~np.isnan(self.t0) \
            and ~np.isnan(self.priwin) and ~np.isnan(self.secwin):
            for s, (t_lst, f_lst) in self.lc_lst.items():
                mask = f_lst < np.percentile(f_lst, 99)
                t1 = t_lst[0]
                t2 = t_lst[-1]
                i1 = int((t1-self.t0)/self.period)
                i2 = int((t2-self.t0)/self.period)
                for i in np.arange(i1, i2+1):
                    _tc = self.t0 + i * self.period
                    _t1 = _tc - self.priwin * self.period
                    _t2 = _tc + self.priwin * self.period
                    _m = (t_lst > _t1)*(t_lst < _t2)
                    mask[_m] = False
                for i in np.arange(i1-1, i2+1):
                    _tc = self.t0 + (i + self.secphase) * self.period
                    _t1 = _tc - self.secwin * self.period
                    _t2 = _tc + self.secwin * self.period
                    _m = (t_lst > _t1)*(t_lst < _t2)
                    mask[_m] = False

                smx, smy = smooth_trend(t_lst[mask], f_lst[mask], self.period/self.detrendwin)
                newf = intp.InterpolatedUnivariateSpline(smx, smy, k=3)
                self.lc_trends[s] = newf(t_lst)

        self.plot()
        self.plot_frame.canvas.draw()

    def defit_ooe(self):
        self.lc_trends = None
        self.plot()
        self.plot_frame.canvas.draw()

    def fit_period(self, option):

        data_lst = self.prepare_parsed_lc()

        def get_disp_lst(data_lst, period, ph1, ph2, nsep):
            disp_lst = []
            dphase = (ph2 - ph1) / nsep
            for i in np.arange(nsep):
                _ph1 = ph1 + i * dphase
                _ph2 = _ph1 + dphase

                allflux_lst = []

                # loop for every sector
                for s, (m, newf_lst) in data_lst.items():
                    t_lst, f_lst = self.lc_lst[s]

                    if not self.sector_mask[s]:
                        continue
                
                    phase_lst = ((t_lst - self.t0)%period)/period
                    # now phase list is between (0, 1)
                    newphase_lst = np.concatenate((phase_lst-1, phase_lst))
                    newflux_lst  = np.concatenate((newf_lst, newf_lst))
                    newm = np.concatenate((m, m))
                    # now phase list is between (-1, 1)
                    m1 = (newphase_lst > _ph1) & (newphase_lst < _ph2)
                    m0 = newm * m1
                    if m0.sum() > 3:
                        for v in newflux_lst[m0]:
                            allflux_lst.append(v)
                
                allflux_lst = np.array(allflux_lst)
                if allflux_lst.size > 0:
                    std = allflux_lst.std()
                    disp_lst.append(std)
            return np.array(disp_lst)

        def get_disp(ratio):
            period = self.period * (1 + ratio)
            nsep = 20
            all_disp_lst = []
            if option == 'primary':
                disp_lst = get_disp_lst(data_lst, period,
                            -2*self.priwin,
                            2*self.priwin,
                            nsep)
                for v in disp_lst:
                    all_disp_lst.append(v)

            elif option == 'secondary':
                disp_lst = get_disp_lst(data_lst, period,
                            self.secphase-2*self.secwin,
                            self.secphase+2*self.secwin,
                            nsep)
                for v in disp_lst:
                    all_disp_lst.append(v)

            elif option == 'both':
                disp_lst = get_disp_lst(data_lst, period,
                            -2*self.priwin,
                            2*self.priwin,
                            nsep)
                for v in disp_lst:
                    all_disp_lst.append(v)
                disp_lst = get_disp_lst(data_lst, period,
                            self.secphase-2*self.secwin,
                            self.secphase+2*self.secwin,
                            nsep)
                for v in disp_lst:
                    all_disp_lst.append(v)

            all_disp_lst = np.array(all_disp_lst)
            disp = (all_disp_lst**2).sum()/all_disp_lst.size
            return disp

        result = opt.minimize_scalar(get_disp,
                    bracket=(-0.01, 0, 0.01), tol=1e-7, method='Brent')
        if result.success:
            newratio = result.x
            new_period = self.period * (1 + newratio)
            # will trigger plot() function here
            self.set_period(new_period)

    def prepare_parsed_lc(self):
        data_lst = {}
        for s in self.sector_lst:

            t_lst, f_lst = self.lc_lst[s]
            m = f_lst < np.percentile(f_lst, 99)

            t_lst, f_lst = self.lc_lst[s]

            # find offset of this sector
            if self.correct_offset:
                offset = self.offset_lst[s]
            else:
                offset = 0.0

            if self.lc_trends is not None:
                trendf = self.lc_trends[s]
                newf_lst = f_lst / trendf * self.medflux_lst[s]
                newf_lst = newf_lst - offset
            else:
                newf_lst = f_lst - offset

            data_lst[s] = (m, newf_lst)
        return data_lst

    
    def fit_eclipse(self, eclipse, model):

        data_lst = self.prepare_parsed_lc()


        if eclipse == 'primary':
            ph1 = -2*self.priwin
            ph2 = 2*self.priwin
        elif eclipse == 'secondary':
            ph1 = self.secphase - 2*self.secwin
            ph2 = self.secphase + 2*self.secwin
        else:
            raise ValueError

        # prepare phase list and allflux_lst to be fitted
        allphase_lst = []
        allflux_lst = []
        for s, (m, newf_lst) in data_lst.items():
            t_lst, f_lst = self.lc_lst[s]

            if not self.sector_mask[s]:
                continue
                
            phase_lst = ((t_lst - self.t0)%self.period)/self.period
            # now phase_lst is between(0,1)
            newphase_lst = np.concatenate((phase_lst-1, phase_lst))
            newflux_lst  = np.concatenate((newf_lst, newf_lst))
            newmask_lst  = np.concatenate((m, m))

            # now above 3 lists between (-1, 1)
            m1 = (newphase_lst > ph1) * (newphase_lst < ph2)
            m0 = newmask_lst * m1

            for _ph, _f, in zip(newphase_lst[m0], newflux_lst[m0]):
                allphase_lst.append(_ph+1)
                allflux_lst.append(_f)

        allphase_lst = np.array(allphase_lst)
        allflux_lst = np.array(allflux_lst)

        # resort
        idx = allphase_lst.argsort()
        allphase_lst = allphase_lst[idx]
        allflux_lst  = allflux_lst[idx]


        # out of elipse points
        f0 = np.percentile(allflux_lst, 75)

        if model == 'kopal':

            inc = np.deg2rad(88)
            u = 0.2

            if eclipse == 'primary':
                phase0 = 0.0
                r1 = self.priwin * 4
                r2 = self.secwin * 4
                p_lower = [0,   -self.priwin, 0.0, 0.0, np.deg2rad(80), 0.0]
                p_upper = [2*f0, self.priwin, 0.5, 0.5, np.deg2rad(90), 1.0]
            elif eclipse == 'secondary':
                phase0 = self.secphase
                r1 = self.secwin * 4
                r2 = self.priwin * 4
                p_lower = [0,    self.secphase-self.secwin, 0.0, 0.0, np.deg2rad(80), 0.0]
                p_upper = [2*f0, self.secphase+self.secwin, 0.5, 0.5, np.deg2rad(90), 1.0]
            else:
                raise ValueError

            p0 = [f0, phase0, r1, r2, inc, u]

            result = opt.least_squares(ecl_errfunc_kopal,
                            x0=p0, bounds=(p_lower, p_upper),
                            args=(allphase_lst, allflux_lst))

            if result.success:
                p = result.x
                f0     = p[0]
                phase0 = p[1]
                r1, r2 = p[2], p[3]
                inc    = p[4]
                u      = p[5]
                
                print('ph0={:10.6f}'.format(phase0))
                print('R1={:10.6f}'.format(r1))
                print('R2={:10.6f}'.format(r2))
                print('inc={:6.3f}'.format(np.rad2deg(inc)))
                print('u={:6.3f}'.format(u))
                
                phase14 = get_kopal_phase14(r1, r2, inc)
                if eclipse == 'primary':
                    # update T0
                    t0 = self.t0 + phase0 * self.period
                    self.set_t0(t0)
                    self.priwin = phase14/2
                    newph_lst = np.linspace(1-2*self.priwin, 1+2*self.priwin, 500)
                    newf_lst = get_ecl_kopal(newph_lst, f0, 0.0, r1, r2, inc, u)
                elif eclipse == 'secondary':
                    self.set_secphase(phase0)
                    self.secwin = phase14/2
                    newph_lst = np.linspace(self.secphase-2*self.secwin, self.secphase+2*self.secwin, 500)
                    newf_lst = get_ecl_kopal(newph_lst, f0, self.secphase, r1, r2, inc, u)
                else:
                    raise ValueError

                self.model_curve[eclipse] = (newph_lst, newf_lst)
                self.plot()
                self.plot_frame.canvas.draw()
        elif model == 'trapz':
            pass
            #depth = np.abs(allphase_lst




    def plot(self):

        fig = self.plot_frame.fig
        fig.clear()

        # add axes
        axlc_lst = []

        plot_sector_lst = [s for s in self.sector_lst
                            if self.sector_mask[s]]
        plot_segment_lst = consecutive(plot_sector_lst)

        # find tspan list
        tspan_lst = []
        for segs in plot_segment_lst:
            s1 = segs[0]
            s2 = segs[-1]
            t1 = self.tspan_lst[s1][0]
            t2 = self.tspan_lst[s2][1]
            tspan_lst.append(t2-t1)
        tspan_lst = np.array(tspan_lst)
        # effective tspan
        eff_tspan = tspan_lst.sum()

        nsector = len(plot_sector_lst)

        gap = 0.006
        ledge = 0.05
        redge = 0.98

        # create GLS axis
        _bottom = 0.08
        _height = 0.39

        axphase = fig.add_axes([ledge, _bottom, 0.238, _height])
        axpri = fig.add_axes([0.305, _bottom, 0.187, _height])
        axsec = fig.add_axes([0.508, _bottom, 0.187, _height])
        axgls = fig.add_axes([0.73, _bottom, redge-0.73, _height])

        # register to class
        self.axphase = axphase
        self.axpri = axpri
        self.axsec = axsec
        self.axgls = axgls

        # calculate width of each axlc
        nseg = len(plot_segment_lst)
        width_gap = (nseg-1)*gap
        width_lst = np.array([(redge - ledge - width_gap)/eff_tspan*tspan
                        for tspan in tspan_lst])

        # loop for segments
        y1_lst, y2_lst = [], []
        for iseg, segs in enumerate(plot_segment_lst):
            s1, s2 = segs[0], segs[-1]
            _left = ledge + width_lst[0:iseg].sum() + iseg*gap
            _width = width_lst[iseg]
            ax = fig.add_axes([_left, 0.51, _width, 0.42])
            axlc_lst.append(ax)

            for s in range(s1, s2+1):

                jseg = self.argseg_lst[s]
                color = 'C{}'.format(jseg)

                t_lst, f_lst = self.lc_lst[s]

                # find the offset of this sector
                if self.correct_offset:
                    offset = self.offset_lst[s]
                else:
                    offset = 0.0

                ax.plot(t_lst, f_lst - offset, 'o', c=color,
                        mew=0, ms=1, alpha=0.8)
                ymax = np.percentile(f_lst - offset, self.display_urej)
                ymin = f_lst.min() - offset

                if self.lc_trends is not None:
                    trend = self.lc_trends[s]
                    indices = np.where(np.diff(t_lst) > 0.5)[0]+1
                    tsplit_lst = np.split(t_lst, indices)
                    _trend_lst = np.split(trend, indices)
                    for _t_lst, _trend in zip(tsplit_lst, _trend_lst):
                        ax.plot(_t_lst, _trend - offset, '-', c='k', lw=0.5)

                yspan = ymax-ymin
                _y1 = ymin - 0.1*yspan
                _y2 = ymax + 0.1*yspan
                y1_lst.append(_y1)
                y2_lst.append(_y2)

                if s == s1:
                    _x1 = t_lst[0]
                if s == s2:
                    _x2 = t_lst[-1]

            # set the boundaries
            ax.set_xlim(_x1, _x2)

            # set the major and minor ticks
            if nsector >= 25 and len(segs)>2:
                major_tick, minor_tick = 50, 10
            if nsector >= 15:
                major_tick, minor_tick = 20, 5
            elif nsector >= 5:
                major_tick, minor_tick = 10, 1
            else:
                major_tick, minor_tick = 5, 1
            ax.xaxis.set_major_locator(tck.MultipleLocator(major_tick))
            ax.xaxis.set_minor_locator(tck.MultipleLocator(minor_tick))
            
            #ax.grid(True, axis='x', ls='--', lw=0.5, alpha=0.6)
            #ax.set_axisbelow(True)

            d = 2
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                        linestyle='none', color='k', mec='k', mew=1, clip_on=False)
            if iseg == 0:
                ax.set_ylabel('PDCSAP_FLUX', fontsize=6)
            if iseg > 0:
                ax.spines.left.set_visible(False)
                ax.tick_params(labelleft=False)
                ax.set_yticks([])
                ax.plot([0, 0], [0, 1], transform=ax.transAxes, **kwargs)
            if iseg < nseg-1:
                ax.spines.right.set_visible(False)
                ax.plot([1, 1], [0, 1], transform=ax.transAxes, **kwargs)

        for iseg, segs in enumerate(self.segment_lst):
            s1, s2 = segs[0], segs[-1]
            # plot the GLS periodogram for the entire segment
            power = self.gls_lst[(s1, s2)]
            color = 'C{}'.format(iseg)
            axgls.plot(self.gls_period_lst, power, '-', lw=0.5, alpha=0.6, c=color)

        # plot overall GLS periodogram
        #axgls.plot(self.gls_period_lst, self.all_gls, '-', lw=0.5, c='k')

        self.axlc_lst = axlc_lst

        # make consistent y1 and y2
        y1 = min(y1_lst)
        y2 = max(y2_lst)
        for iseg, ax in enumerate(axlc_lst):
            ax.set_ylim(y1, y2)
            # display sector range for this axes
            segs = plot_segment_lst[iseg]
            s1 = segs[0]
            s2 = segs[-1]
            x1, x2 = ax.get_xlim()
            if len(segs)==1:
                text = 'S{:02d}'.format(s1)
            else:
                text = 'S{:02d}-{:02d}'.format(s1, s2)
            ax.text(0.97*x1+0.03*x2, 0.08*y1+0.92*y2, text, fontsize=6)
            # set x and y tick font
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(6)


        # plot small ticks indicating the time of primary and secondary
        # eclipses
        if self.period is not np.nan and self.t0 is not np.nan:
            for iseg, segs in enumerate(plot_segment_lst):
                ax = axlc_lst[iseg]
                s1, s2 = segs[0], segs[-1]
                t1, t2 = ax.get_xlim()

                i1 = int((t1 - self.t0)/self.period)
                i2 = int((t2 - self.t0)/self.period)
                for i in np.arange(i1, i2+1):
                    _t = self.t0 + i * self.period
                    ax.axvline(_t, 0, 1, c='r', ls='-',
                            lw=0.5, alpha=0.2, zorder=-1)
                if self.secphase is not np.nan:
                    for i in np.arange(i1-1, i2+1):
                        _t = self.t0 + (i + self.secphase) * self.period
                        ax.axvline(_t, 0, 1, c='b', ls='-',
                                lw=0.5, alpha=0.2, zorder=-1)


        # plot determind period
        if self.period is not np.nan:
            axgls.axvline(self.period, color='k', lw=0.7, ls='--')
        # set the axes of GLS periodogram
        axgls.set_xscale('log')
        axgls.set_yscale('log')
        axgls.set_xlim(0.1, 100)
        axgls.set_xlabel('Period (day)', fontsize=6)
        #axgls.axvline(1440/2, c='k', ls='--', lw=0.5)
        # set tick font and xlabel
        for ax in [axphase, axpri, axsec, axgls]:
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
        for ax in [axphase, axpri, axsec]:
            ax.set_xlabel(u'\u03c6', fontsize=6)
        axphase.set_ylabel('PDCSAP_FLUX', fontsize=6)

        # plot phase folded diagrams
        if ~np.isnan(self.period):

            for segs in plot_segment_lst:
                for s in segs:

                    t_lst, f_lst = self.lc_lst[s]
                    phase_lst = ((t_lst-self.t0)%self.period)/self.period
                    # determine color
                    jseg = self.argseg_lst[s]
                    color = 'C{}'.format(jseg)

                    # determin y offset
                    if self.correct_offset:
                        _offset = self.offset_lst[s]
                    else:
                        _offset = 0.0

                    if self.lc_trends is not None:
                        trendf = self.lc_trends[s]
                        newf_lst = f_lst / trendf * self.medflux_lst[s]
                    else:
                        newf_lst = f_lst

                    axphase.plot(phase_lst, newf_lst - _offset, 'o', color=color,
                                mew=0, ms=1, alpha=0.6)
                    axphase.plot(phase_lst+1, newf_lst - _offset, 'o', color=color,
                                mew=0, ms=1, alpha=0.6)
                    axpri.plot(phase_lst, newf_lst - _offset, 'o', color=color,
                                mew=0, ms=1, alpha=0.6)
                    axpri.plot(phase_lst+1, newf_lst - _offset, 'o', color=color,
                                mew=0, ms=1, alpha=0.6)
                    axsec.plot(phase_lst, newf_lst - _offset, 'o', color=color,
                                mew=0, ms=1, alpha=0.6)

        
        if self.model_curve['primary'] is not None:
            phase_lst, flux_lst = self.model_curve['primary']
            axpri.plot(phase_lst, flux_lst, '-', lw=0.5, c='k', alpha=0.8)
        if self.model_curve['secondary'] is not None:
            phase_lst, flux_lst = self.model_curve['secondary']
            axsec.plot(phase_lst, flux_lst, '-', lw=0.5, c='k', alpha=0.8)
       
        # set axpri range
        axpri.set_xlim(1-self.priwin*2, 1+self.priwin*2)
        axsec.set_xlim(self.secphase-self.secwin*2, self.secphase+self.secwin*2)

        all_t_lst, all_f_lst = self.all_lc
        _y1, _y2 = axpri.get_ylim()
        v99 = np.percentile(all_f_lst, 99)
        _y2 = min(_y2, v99)
        for ax in [axphase, axpri, axsec]:
            ax.set_ylim(_y1, _y2+(_y2-_y1)*0.1)
        axphase.set_xlim(0, 2)


        # plot axvlins in axpri
        ylim = axphase.get_ylim()
        axphase.fill_betweenx(ylim, 1-self.priwin,  1+self.priwin,
                color='r', alpha=0.1, lw=0)
        axphase.fill_betweenx(ylim, 0, self.priwin,
                color='r', alpha=0.1, lw=0)
        axphase.fill_betweenx(ylim, 2-self.priwin, 2,
                color='r', alpha=0.1, lw=0)
        axphase.fill_betweenx(ylim,
                self.secphase-self.secwin,  self.secphase+self.priwin,
                color='b', alpha=0.1, lw=0)
        axphase.fill_betweenx(ylim,
                1+self.secphase-self.secwin,  1+self.secphase+self.secwin,
                color='b', alpha=0.1, lw=0)
        axpri.axvline(1.0, c='r', ls='-', lw=0.5, alpha=0.1, zorder=-1)
        axpri.fill_betweenx(ylim, 1-self.priwin,  1+self.priwin,
                color='r', alpha=0.1, lw=0, zorder=-2)
        axsec.axvline(self.secphase, c='b', ls='-', lw=0.5, alpha=0.1, zorder=-1)
        axsec.fill_betweenx(ylim,
                self.secphase-self.secwin,  self.secphase+self.secwin,
                color='b', alpha=0.1, lw=0, zorder=-2)
        axphase.set_ylim(ylim)
        axpri.set_ylim(ylim)
        axsec.set_ylim(ylim)


        axpri.set_yticklabels([])
        axsec.set_yticklabels([])

        # add period label
        label = 'P = {:.6f} d'.format(self.period)
        _x = 0.07
        _y = 0.9*_y1 + 0.1*_y2
        axphase.text(_x, _y, label, va='bottom', ha='left', fontsize=6)
        # add plotting information
        n_sector = len(self.sector_lst)
        n_plotted = sum(list(self.sector_mask.values()))
        n_hidden = n_sector - n_plotted
        info = '{}, {} plotted, {} hidden'.format(
                '1 sector' if n_sector==1 else '{} sectors'.format(n_sector),
                n_plotted, n_hidden)
        _y = 0.96*_y1 + 0.04*_y2
        axphase.text(_x, _y, info, va='bottom', ha='left', fontsize=5)

        # add title
        title = 'TIC {} '.format(self.tic)
        fig.suptitle(title, fontsize=9)

class PlotFigure(Figure):

    def __init__(self, width, height, dpi):
        figsize  = (width/dpi, height/dpi)
        super(PlotFigure, self).__init__(figsize=figsize, dpi=dpi)


class PlotFrame(tk.Frame):
    def __init__(self, master, width, height, dpi):
        self.master = master
        tk.Frame.__init__(self, master, width=width, height=height)

        self.fig = PlotFigure(width = width,
                              height = height,
                              dpi    = dpi,
                              )
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.control_panel = ControlPanel(master=self)
        self.control_panel.pack(side=tk.TOP, pady=5)

        self.pack()


    def on_click(self, event):
        """Event handler for clicking in the figure.
        """

        if not hasattr(self.master, 'axgls'):
            # figure has not been plotted yet
            return

        if event.inaxes == self.master.axgls:
            # click on the GLS figure
            if event.dblclick:
                p0 = event.xdata
                i = np.searchsorted(self.master.gls_period_lst, p0)
                i1 = i - 5
                i2 = i + 5
                imax = self.master.all_gls[i1:i2].argmax() + i1
                maxperiod = self.master.gls_period_lst[imax]
                self.master.set_period(maxperiod)

        else:
            if event.dblclick:
                for ax in self.master.axlc_lst:
                    if event.inaxes == ax:
                        # clicked time
                        t0 = event.xdata
                        for sector, (t_lst, f_lst) in self.master.lc_lst.items():
                            if t_lst[0] < t0 < t_lst[-1]:
                                # find scale of this axes
                                # scale = time/(fraction of figure)
                                bbox = ax.get_position()
                                width = bbox.width
                                _t1, _t2 = ax.get_xlim()
                                scale = (_t2 - _t1)/width
                                # set search range to +/- 0.5% of figure width
                                dt = 0.005 * scale
                                t1 = t0 - dt
                                t2 = t0 + dt
                                i1 = np.searchsorted(t_lst, t1)
                                i2 = np.searchsorted(t_lst, t2)
                                # find the local minima
                                imin = f_lst[i1:i2].argmin() + i1
                                tmin = t_lst[imin]
                                # call set_t0 function to change t0 and replot
                                self.master.set_t0(tmin)
                                break

class ControlPanel(tk.Frame):
    def __init__(self, master):
        self.master = master
        tk.Frame.__init__(self, master, width=400)

        #self.grid_rowconfigure(0, weight=1)
        #self.grid_rowconfigure(1, weight=1)
        #self.grid_rowconfigure(2, weight=1)
        #self.grid_rowconfigure(3, weight=1)
        #self.grid_rowconfigure(4, weight=1)
        #self.grid_rowconfigure(5, weight=1)
        #self.grid_columnconfigure(0, weight=1)
        #self.grid_columnconfigure(1, weight=1)
        #self.grid_columnconfigure(2, weight=1)
        #self.grid_columnconfigure(4, weight=1)


        icol = 0
        ###### set column 0 ############
        self.base_flux_label = tk.Label(self,
                    text='Base Flux', font=('TkDefaultFont', 11))
        self.urej_label = tk.Label(self,
                    text='Displayed Upper Rej. =', font=('TkDefaultFont', 11))
        self.urej = tk.IntVar(value=100)
        self.urej_spinbox = tk.Spinbox(self,
                    from_        = 50,
                    to           = 100,
                    width        = 4,
                    textvariable = self.urej,
                    command      = self.adjust_urej,
                    state        = tk.DISABLED,
                    wrap         = False,
                    )
        self.correct_offset = tk.BooleanVar(value=True)
        self.offset_cb = tk.Checkbutton(self,
                    text     = 'Correct Flux Offset',
                    variable = self.correct_offset,
                    command  = self.adjust_offset,
                    font     = ('TkDefaultFont', 11),
                    state    = tk.DISABLED,
                    )


        icol = 0
        self.base_flux_label.grid(row=0, column=icol, columnspan=2,
                sticky=tk.EW)
        self.urej_label.grid(row=1, column=icol,
                sticky=tk.EW, padx=5, pady=2)
        self.urej_spinbox.grid(row=1, column=icol+1,
                sticky=tk.EW, padx=5, pady=2)
        self.offset_cb.grid(row=2, column=icol, columnspan=2,
                sticky=tk.EW, padx=5, pady=2)


        ############## add a vertical separator
        icol += 2
        sepv0 = ttk.Separator(self, orient='vertical')
        sepv0.grid(row=1, column=icol, rowspan=6, sticky='ns', padx=10)



        ############ set column 1 ############
        self.period_label = tk.Label(self,
                    text='Period', font=('TkDefaultFont', 11))

        #############################
        self.period_half_button = tk.Button(self,
                    text='Period x 1/2', width=15, state = tk.DISABLED,
                    command = lambda: self.change_period(0.5))
        self.period_double_button = tk.Button(self,
                    text='Period x 2', width=15, state = tk.DISABLED,
                    command = lambda: self.change_period(2))
                                                
        ######### period +1-1####
        self.sub_period_buttons = {
                1: tk.Button(self, text='-', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(0.99)),
                2: tk.Button(self, text='-', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(0.999)),
                3: tk.Button(self, text='-', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(0.9999)),
                4: tk.Button(self, text='-', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(0.99999)),
                5: tk.Button(self, text='-', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(0.999999)),
                }
        self.add_period_buttons = {
                1: tk.Button(self, text='+', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(1.01)),
                2: tk.Button(self, text='+', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(1.001)),
                3: tk.Button(self, text='+', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(1.0001)),
                4: tk.Button(self, text='+', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(1.00001)),
                5: tk.Button(self, text='+', width=5, state=tk.DISABLED,
                        command = lambda: self.change_period(1.000001)),
                }

        label_font = ('TkDefaultFont', 10)
        period_level_labels = {
                1: tk.Label(self, font=label_font, anchor='e',
                    text = u'1% \xd7 Period'),
                2: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.1% \xd7 Period'),
                3: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.01% \xd7 Period'),
                4: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.001% \xd7 Period'),
                5: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.0001% \xd7 Period'),
                }

        ######## button layout ###########
        icol += 1
        self.period_label.grid(row=0, column=icol, columnspan=4, sticky=tk.EW)
        self.period_half_button.grid(row=1, column=icol, columnspan=2,
                                    sticky=tk.EW, padx=5, pady=2)
        self.period_double_button.grid(row=1, column=icol+2, columnspan=2,
                                    sticky=tk.EW, padx=5, pady=2)

        for key, button in self.sub_period_buttons.items():
            button.grid(row=key+1, column=icol, sticky='w', padx=5, pady=2)
        for key, button in self.add_period_buttons.items():
            button.grid(row=key+1, column=icol+3, sticky='e', padx=5, pady=2)
        for key, label in period_level_labels.items():
            label.grid(row=key+1, column=icol+1, columnspan=2,
                    sticky='ew', padx=28, pady=2)

        icol += 4
        ############## add a vertical separator
        sepv1 = ttk.Separator(self, orient='vertical')
        sepv1.grid(row=1, column=icol, rowspan=6, sticky='ns', padx=10)


        ######## set column 2 #####################
        self.t0_label = tk.Label(self,
                    text='T0', font=('TkDefaultFont', 11))
        self.t0_subhalf_button = tk.Button(self,
                text='T0 - 1/2 Period', width=15, state=tk.DISABLED,
                command=lambda: self.change_t0(-0.5))
        self.t0_addhalf_button = tk.Button(self,
                text='T0 + 1/2 Period', width=15, state=tk.DISABLED,
                command=lambda: self.change_t0(+0.5))

        self.add_t0_buttons = {
                1: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(+0.1)),
                2: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(+0.01)),
                3: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(+0.001)),
                4: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(+0.0001)),
                }
        self.sub_t0_buttons = {
                1: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(-0.1)),
                2: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(-0.01)),
                3: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(-0.001)),
                4: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_t0(-0.0001)),
                }

        t0_level_labels = {
                1: tk.Label(self, font=label_font, anchor='e',
                    text = u'10% \xd7 Period'),
                2: tk.Label(self, font=label_font, anchor='e',
                    text = u'1% \xd7 Period'),
                3: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.1% \xd7 Period'),
                4: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.01% \xd7 Period'),
                }
        ####################
        icol += 1

        self.t0_label.grid(row=0, column=icol, columnspan=4, sticky=tk.EW)
        self.t0_subhalf_button.grid(row=1, column=icol, columnspan=2,
                                    sticky=tk.EW, padx=5, pady=2)
        self.t0_addhalf_button.grid(row=1, column=icol+2, columnspan=2,
                                    sticky=tk.EW, padx=5, pady=2)

        for key, button in self.add_t0_buttons.items():
            button.grid(row=key+1, column=icol, sticky='w', padx=5, pady=2)
        for key, button in self.sub_t0_buttons.items():
            button.grid(row=key+1, column=icol+3, sticky='e', padx=5, pady=2)
        for key, label in t0_level_labels.items():
            label.grid(row=key+1, column=icol+1, columnspan=2,
                    sticky='ew', padx=28, pady=2)

        ############## add a vertical separator
        icol += 4

        sepv2 = ttk.Separator(self, orient='vertical')
        sepv2.grid(row=1, column=icol, rowspan=6, sticky='ns', padx=10)



        ################
        self.secphase_label = tk.Label(self,
                text=u'\u03c6 (sec)', font=('TKDefualtFont', 11))

        self.add_secphase_buttons = {
                1: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(+0.1)),
                2: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(+0.01)),
                3: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(+0.001)),
                4: tk.Button(self, text ='<', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(+0.0001)),
                }
        self.sub_secphase_buttons = {
                1: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(-0.1)),
                2: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(-0.01)),
                3: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(-0.001)),
                4: tk.Button(self, text ='>', width=5, state=tk.DISABLED,
                        command=lambda: self.change_secphase(-0.0001)),
                }
        secphase_level_labels = {
                1: tk.Label(self, font=label_font, anchor='e',
                    text = u'10% \xd7 Period'),
                2: tk.Label(self, font=label_font, anchor='e',
                    text = u'1% \xd7 Period'),
                3: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.1% \xd7 Period'),
                4: tk.Label(self, font=label_font, anchor='e',
                    text = u'0.01% \xd7 Period'),
                }

        ###### layout
        icol += 1
        self.secphase_label.grid(row=0, column=icol, columnspan=3, sticky='ew')
        for key, button in self.add_secphase_buttons.items():
            button.grid(row=key, column=icol, sticky='w', padx=5, pady=2)
        for key, button in self.sub_secphase_buttons.items():
            button.grid(row=key, column=icol+2, sticky='e', padx=5, pady=2)
        for key, label in secphase_level_labels.items():
            label.grid(row=key, column=icol+1, sticky='ew', padx=28, pady=2)

        ############## add a vertical separator
        icol += 3
        sepv3 = ttk.Separator(self, orient='vertical')
        sepv3.grid(row=1, column=icol, rowspan=6, sticky='ns', padx=10)

        ########### set column 3 #############################
        self.zoomout_priwin_button = tk.Button(self,
                            text=u'\u2296', width=5, state=tk.DISABLED,
                            command = lambda: self.change_priwin(1.414))
        self.zoomin_priwin_button = tk.Button(self,
                            text=u'\u2295', width=5, state=tk.DISABLED,
                            command = lambda: self.change_priwin(0.707))
        priwin_label = tk.Label(self, font=label_font, justify='center',
                                text='Primary Window')
        self.zoomout_secwin_button = tk.Button(self,
                            text=u'\u2296', width=5, state=tk.DISABLED,
                            command = lambda: self.change_secwin(1.414))
        self.zoomin_secwin_button = tk.Button(self,
                            text=u'\u2295', width=5, state=tk.DISABLED,
                            command = lambda: self.change_secwin(0.707))
        secwin_label = tk.Label(self, font=label_font, justify='center',
                                text='Secondary Window')

        self.detrend_label = tk.Label(self,
                            text='Detrend', font=('TkDefualtFont', 11))
        self.detrend_win = tk.IntVar(value=20)
        self.detrend_spinbox = tk.Spinbox(self,
                            from_        = 10,
                            to           = 100,
                            width        = 1,
                            textvariable = self.detrend_win,
                            command      = self.adjust_detrend_win,
                            state        = tk.DISABLED,
                            wrap         = False,
                            )
        self.fit_ooe_button = tk.Button(self,
                            text='Fit OoE', width=10, state=tk.DISABLED,
                            command = lambda: self.fit_ooe())

        self.autoperiod_button = tk.Button(self,
                            text='Auto Period', width=10, state=tk.DISABLED,
                            command = lambda: self.fit_period_auto())
        self.autoperiod_options = tk.StringVar(value='primary')
        self.autoperiod_rbs = {
                'primary': tk.Radiobutton(self,
                            text     = 'Primary',
                            variable = self.autoperiod_options,
                            value    = 'primary',
                            state    = tk.DISABLED,
                            ),
                'secondary': tk.Radiobutton(self,
                            text     = 'Secondary',
                            variable = self.autoperiod_options,
                            value    = 'secondary',
                            state    = tk.DISABLED,
                            ),
                'both':      tk.Radiobutton(self,
                            text     = 'Both',
                            variable = self.autoperiod_options,
                            value    = 'both',
                            state    = tk.DISABLED,
                            ),
                }

        self.fitpri_button = tk.Button(self,
                            text='Fit Pri.', width=8, state=tk.DISABLED,
                            command = lambda: self.fit_primary())
        self.fitsec_button = tk.Button(self,
                            text='Fit Sec.', width=8, state=tk.DISABLED,
                            command = lambda: self.fit_secondary())
        self.model_pri = tk.StringVar(value='kopal')
        self.model_sec = tk.StringVar(value='kopal')
        self.model_pri_rbs = {
                'kopal': tk.Radiobutton(self,
                            text     = 'Kopal',
                            variable = self.model_pri,
                            value    = 'kopal',
                            state    = tk.DISABLED,
                            ),
                'trapz': tk.Radiobutton(self,
                            text     = 'Trapezoid',
                            variable = self.model_pri,
                            value    = 'trapz',
                            state    = tk.DISABLED,
                            ),
                }
        self.model_sec_rbs = {
                'kopal': tk.Radiobutton(self,
                            text     = 'Kopal',
                            variable = self.model_sec,
                            value    = 'kopal',
                            state    = tk.DISABLED,
                            ),
                'trapz': tk.Radiobutton(self,
                            text     = 'Trapezoid',
                            variable = self.model_sec,
                            value    = 'trapz',
                            state    = tk.DISABLED,
                            ),
                }


        ################################
        icol += 1

        self.zoomout_priwin_button.grid(row=1, column=icol,
                            sticky='w', padx=5, pady=2)
        self.zoomin_priwin_button.grid(row=1, column=icol+3,
                            sticky='e', padx=5, pady=2)
        self.zoomout_secwin_button.grid(row=2, column=icol,
                            sticky='w', padx=5, pady=2)
        self.zoomin_secwin_button.grid(row=2, column=icol+3,
                            sticky='e', padx=5, pady=2)
        priwin_label.grid(row=1, column=icol+1, columnspan=2,
                            sticky='ew', padx=20, pady=2)
        secwin_label.grid(row=2, column=icol+1, columnspan=2,
                            sticky='ew', padx=20, pady=2)

        self.detrend_label.grid(row=3, column=icol,
                            sticky='w', padx=5, pady=2)
        self.detrend_spinbox.grid(row=3, column=icol+1,
                            sticky='ew', padx=5, pady=2)
        self.fit_ooe_button.grid(row=3, column=icol+2, columnspan=2,
                            sticky='ew', padx=5, pady=2)
        self.autoperiod_button.grid(row=4, column=icol, columnspan=1,
                            sticky='ew', padx=5, pady=2)
        for icb, option in enumerate(['primary', 'secondary', 'both']):
            self.autoperiod_rbs[option].grid(row=4, column=icol+icb+1,
                            sticky='w', padx=5, pady=2)

        self.fitpri_button.grid(row=5, column=icol, columnspan=1,
                            sticky='ew', padx=5, pady=2)
        self.fitsec_button.grid(row=6, column=icol, columnspan=1,
                            sticky='ew', padx=5, pady=2)
        for icb, modelname in enumerate(['kopal', 'trapz']):
            self.model_pri_rbs[modelname].grid(row=5, column=icol+icb+1,
                            columnspan=2, sticky='w', padx=5, pady=2)
            self.model_sec_rbs[modelname].grid(row=6, column=icol+icb+1,
                            columnspan=2, sticky='w', padx=5, pady=2)

        #sep = ttk.Separator(self, orient='horizontal')
        #sep.grid(row=3, column=icol, columnspan=3, sticky='ew', padx=5, pady=2)

        icol += 4
        ############## add a vertical separator
        sepv4 = ttk.Separator(self, orient='vertical')
        sepv4.grid(row=1, column=icol, rowspan=6, sticky='ns', padx=10)
        icol += 1


        ####### 
        self.apply_button = tk.Button(self,
                                text  = 'Apply',
                                width = 15,
                                state = tk.DISABLED,
                                command = self.apply_params,
                                )

        self.subtype = tk.StringVar(value='detached')
        self.subtype_rbs = {
                'detached': tk.Radiobutton(self,
                                    text     = 'Detached',
                                    variable = self.subtype,
                                    value    = 'detached',
                                    state    = tk.DISABLED,
                                    ),
                'contact': tk.Radiobutton(self,
                                    text     = 'Contact',
                                    variable = self.subtype,
                                    value    = 'contact',
                                    state    = tk.DISABLED,
                                    ),
                }


        # create a boolean variable for each flag 
        self.flags = {flag: tk.BooleanVar(value=False)
                        for flag, text in self.master.master.flag_lst}

        # create a check button for each flag
        self.flag_cbs = {flag: tk.Checkbutton(self,
                                    text     = text,
                                    variable = self.flags[flag],
                                    font     = ('TkDefaultFont', 11),
                                    state    = tk.DISABLED,
                                    onvalue  = True,
                                    offvalue = False,
                                    )
                            for flag, text in self.master.master.flag_lst
                            }


        self.apply_button.grid(row=0, column=icol, columnspan=2,
                sticky='ew', padx=5, pady=2)
        for icb, subtype in enumerate(['detached', 'contact']):
            self.subtype_rbs[subtype].grid(row=1, column=icol+icb,
                    sticky='ew', padx=5, pady=2)
        for i, (flag, cb) in enumerate(self.flag_cbs.items()):
            cb.grid(row=2+i, column=icol, columnspan=2,
                    sticky='w', padx=5, pady=2)

        self.pack()

    def change_period(self, ratio):
        self.master.master.change_period(ratio)

    def change_t0(self, level):
        self.master.master.change_t0(level)

    def change_priwin(self, ratio):
        self.master.master.change_priwin(ratio)

    def change_secwin(self, ratio):
        self.master.master.change_secwin(ratio)

    def change_secphase(self, ratio):
        self.master.master.change_secphase(ratio)

    def adjust_urej(self):
        self.master.master.change_urej(self.urej.get())

    def adjust_offset(self):
        self.master.master.change_offset(self.correct_offset.get())

    def adjust_detrend_win(self):
        self.master.master.change_detrendwin(self.detrend_win.get())

    def fit_ooe(self):

        # judge the state according to the lc_trends=None
        if self.master.master.lc_trends is None:
            # will trigger the plot() function
            self.master.master.fit_ooe()
            # change the text of fit button
            self.fit_ooe_button['text'] = 'DeFit'
        else:
            # will trigger the plot() function
            self.master.master.defit_ooe()
            # change the text of fit button
            self.fit_ooe_button['text'] = 'Fit OoE'

    def fit_period_auto(self):
        option = self.autoperiod_options.get()
        self.master.master.fit_period(option)

    def fit_primary(self):
        model = self.model_pri.get()
        self.master.master.fit_eclipse('primary', model)

    def fit_secondary(self):
        model = self.model_sec.get()
        self.master.master.fit_eclipse('secondary', model)

    def set_button(self, state):
        if state:

            self.urej_spinbox['state'] = tk.NORMAL
            self.offset_cb['state'] = tk.NORMAL
            # period control buttons
            self.period_double_button['state'] = tk.NORMAL
            self.period_half_button['state']   = tk.NORMAL
            for key, button in self.sub_period_buttons.items():
                button['state'] = tk.NORMAL
            for key, button in self.add_period_buttons.items():
                button['state'] = tk.NORMAL

            # T0 buttons
            self.t0_subhalf_button['state']  = tk.NORMAL
            self.t0_addhalf_button['state']  = tk.NORMAL
            for key, button in self.add_t0_buttons.items():
                button['state'] = tk.NORMAL
            for key, button in self.sub_t0_buttons.items():
                button['state'] = tk.NORMAL

            # window zoomin/out buttons
            self.zoomin_priwin_button['state'] = tk.NORMAL
            self.zoomout_priwin_button['state'] = tk.NORMAL
            self.zoomin_secwin_button['state'] = tk.NORMAL
            self.zoomout_secwin_button['state'] = tk.NORMAL

            self.detrend_spinbox['state'] = tk.NORMAL
            self.fit_ooe_button['state'] = tk.NORMAL
            self.autoperiod_button['state'] = tk.NORMAL
            for key, rb in self.autoperiod_rbs.items():
                rb['state'] = tk.NORMAL

            self.fitpri_button['state'] = tk.NORMAL
            self.fitsec_button['state'] = tk.NORMAL
            for key, rb in self.model_pri_rbs.items():
                rb['state'] = tk.NORMAL
            for key, rb in self.model_sec_rbs.items():
                rb['state'] = tk.NORMAL

            # secondary phase buttons
            for key, button in self.add_secphase_buttons.items():
                button['state'] = tk.NORMAL
            for key, button in self.sub_secphase_buttons.items():
                button['state'] = tk.NORMAL


            # apply button
            self.apply_button['state'] = tk.NORMAL
            self.apply_button.configure(
                    bg='#4D90FE', fg='white',
                    activebackground='#357AE8',
                    activeforeground='white',
                    )
            # subtype radio buttons
            for subtype, rb in self.subtype_rbs.items():
                rb['state'] = tk.NORMAL
            # falg check buttons
            for flag, cb in self.flag_cbs.items():
                cb['state'] = tk.NORMAL

    def update_param(self):
        text = 'Period = {:.7f} d'.format(self.master.master.period)
        self.period_label.config(text=text)

        text = 'T0 = {:.7f}'.format(self.master.master.t0)
        self.t0_label.config(text=text)

        text = u'\u03c6 (sec) = {:.4f}'.format(self.master.master.secphase)
        self.secphase_label.config(text=text)

    def reset_param(self):
        # reset upper rejection spinbox
        self.urej.set(self.master.master.display_urej)
        # reset correct-offset check button
        self.correct_offset.set(self.master.master.correct_offset)

        # reset fit buttons
        self.fit_ooe_button['text'] = 'Fit OoE'

        # reset subtype and flags
        self.subtype.set('detached')
        for flag, var in self.flags.items():
            var.set(False)

    def apply_params(self):
        self.master.master.right_frame.source_frame.apply_params()

class RightFrame(tk.Frame):
    def __init__(self, master, width, height, source_filename):
        self.master = master
        tk.Frame.__init__(self, master, width=width, height=height)

        self.source_frame = SourceFrame(master = self,
                                        width  = width,
                                        height = height-100,
                                        source_filename = source_filename,
                                        )
        self.sector_frame = SectorFrame(master = self,
                                        width  = width,
                                        height = 500,
                                        )


class SourceFrame(tk.Frame):
    def __init__(self, master, width, height, source_filename):

        self.master = master
        self.source_filename = source_filename

        self.source_table = Table.read(source_filename,
                            format='ascii.fixed_width_two_line')
        n = len(self.source_table)

        if 'base_flux' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='base_flux',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'Period' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='Period',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'T0' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='T0',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'tdur_pri' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='tdur_pri',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'tdur_sec' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='tdur_sec',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'phi_sec' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='phi_sec',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'sector1' not in self.source_table.colnames:
            column = MaskedColumn([0]*n, name='sector1',
                            mask=[True]*n, dtype=int)
            self.source_table.add_column(column, index=-1)

        if 'sector2' not in self.source_table.colnames:
            column = MaskedColumn([0]*n, name='sector2',
                            mask=[True]*n, dtype=int)
            self.source_table.add_column(column, index=-1)

        if 'nsector' not in self.source_table.colnames:
            column = MaskedColumn([0]*n, name='nsector',
                            mask=[True]*n, dtype=int)
            self.source_table.add_column(column, index=-1)

        if 'tspan' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='tspan',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'tlength' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='tlength',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'npri' not in self.source_table.colnames:
            column = MaskedColumn([0]*n, name='npri',
                            mask=[True]*n, dtype=int)
            self.source_table.add_column(column, index=-1)

        if 'nsec' not in self.source_table.colnames:
            column = MaskedColumn([0]*n, name='nsec',
                            mask=[True]*n, dtype=int)
            self.source_table.add_column(column, index=-1)

        if 'tspan_pri' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='tspan_pri',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'tspan_sec' not in self.source_table.colnames:
            column = MaskedColumn([0.0]*n, name='tspan_sec',
                            mask=[True]*n, dtype='f8')
            self.source_table.add_column(column, index=-1)

        if 'subtype' not in self.source_table.colnames:
            column = MaskedColumn(['']*n, name='subtype',
                            mask=[True]*n, dtype='S8')
            self.source_table.add_column(column, index=-1)

        for flag, text in master.master.flag_lst:
            if flag not in self.source_table.colnames:
                column = MaskedColumn(['']*n, name=flag,
                                mask=[True]*n, dtype='S1')
                self.source_table.add_column(column, index=-1)
            

        tk.Frame.__init__(self, master, width=width, height=height)
        

        self.save_button = tk.Button(master = self,
                                    text    = 'Save List',
                                    width   = 150,
                                    command = self.save_source_table,
                                    state   = tk.DISABLED,
                                    )

        columns = ('tic', 'period', 'phisec', 'subtype', 'notes')
        self.source_tree = ttk.Treeview(self,
                columns = columns,
                show    = 'headings',
                style   = 'Treeview',
                height  = 20,
                selectmode = 'browse',
                )

        self.source_tree.bind('<<TreeviewSelect>>', self.on_select_item)

        self.scrollbar = tk.Scrollbar(self,
                orient = tk.VERTICAL,
                width  = 10,
                )

        self.source_tree.column('tic', width=90, anchor='e')
        self.source_tree.column('period', width=100, anchor='e')
        #self.source_tree.column('t0', width=120, anchor='e')
        #self.source_tree.column('tpri', width=60, anchor='e')
        #self.source_tree.column('tsec', width=60, anchor='e')
        self.source_tree.column('phisec', width=80, anchor='e')
        self.source_tree.column('subtype', width=80, anchor='e')
        self.source_tree.column('notes', width=width-370, anchor='w')

        self.source_tree.heading('tic',    text='TIC')
        self.source_tree.heading('period', text='Period')
        #self.source_tree.heading('t0',     text='T0')
        #self.source_tree.heading('tpri',   text='t (pri)')
        #self.source_tree.heading('tsec',   text='t (sec)')
        self.source_tree.heading('phisec', text=u'\u03c6 (sec)')
        self.source_tree.heading('subtype',text='subtype')
        self.source_tree.heading('notes',  text='notes')

        self.source_tree.config(yscrollcommand = self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=30)

        self.scrollbar.config(command=self.source_tree.yview)

        for row in self.source_table:
            tic = row['TIC']
            period = '{:10.6f}'.format(row['Period']) if row['Period'] is not np.ma.masked else ''
            #t_pri = row['t_pri'] if row['t_pri'] is not np.ma.masked else ''
            #t_sec = row['t_sec'] if row['t_sec'] is not np.ma.masked else ''
            phi_sec = '{:6.4f}'.format(row['phi_sec']) if row['phi_sec'] is not np.ma.masked else ''
            subtype = row['subtype'] if row['subtype'] is not np.ma.masked else ''

            tag_lst = []
            for flag, text in self.master.master.flag_lst:
                if row[flag] is not np.ma.masked and row[flag]==1:
                    tag_lst.append(flag.upper())
            notes = ', '.join(tag_lst)

            item = (tic, period, phi_sec, subtype, notes)
            iid = self.source_tree.insert('', tk.END,
                        values=item, tags='normal', open=True)

        self.save_button.pack(side=tk.TOP)
        self.source_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        self.pack()

    def on_select_item(self, event):

        items = event.widget.selection()
        if len(items)==0:
            return None

        item = items[0]
        values = self.source_tree.item(item, 'values')

        # reset parameters
        self.master.master.reset_param()

        tic = int(values[0])
        m = self.source_table['TIC']==tic
        row = self.source_table[m][0]

        # set aliases
        mainwin = self.master.master
        control_panel = mainwin.plot_frame.control_panel

        mainwin.tic = tic
        if row['Period'] is not np.ma.masked:
            mainwin.period = row['Period']
            if row['tdur_pri'] is not np.ma.masked:
                mainwin.priwin = row['tdur_pri']/2/row['Period']
            if row['tdur_sec'] is not np.ma.masked:
                mainwin.secwin = row['tdur_sec']/2/row['Period']
        if row['T0'] is not np.ma.masked:
            mainwin.t0 = row['T0']
        if row['phi_sec'] is not np.ma.masked:
            mainwin.secphase = row['phi_sec']


        # read light curves
        mainwin.read_lc()

        # put sector list into the sector tree
        self.master.sector_frame.load_sectors(mainwin.sector_lst)

        # find sector-by-sector offsets
        mainwin.find_offset()
        # prepare the segmented light curves
        mainwin.get_seg_lc()
        # calculate GLS periodogram of each segment
        mainwin.calc_gls()

        if np.isnan(mainwin.period):
            mainwin.find_period_auto()
        if np.isnan(mainwin.t0):
            mainwin.find_t0_auto()


        mainwin.plot()
        mainwin.plot_frame.canvas.draw()

        control_panel.reset_param()
        control_panel.set_button(True)
        # refresh parameters in the control panel
        control_panel.update_param()

        # check the source table, and set the subtypes
        if row['subtype'] is not np.ma.masked \
            and row['subtype'] in control_panel.subtype_rbs:
            control_panel.subtype.set(row['subtype'])

        # check the source table, and toggle the flag check buttons
        for flag, var in control_panel.flags.items():
            if row[flag] is not np.ma.masked and row[flag]==1:
                control_panel.flag_cbs[flag].select()


    def apply_params(self):

        mainwin  = self.master.master

        # alias for control panel
        control_panel = mainwin.plot_frame.control_panel
        # set parameter aliases
        tic      = mainwin.tic
        medflux  = mainwin.medflux
        period   = mainwin.period
        t0       = mainwin.t0
        secphase = mainwin.secphase
        priwin   = mainwin.priwin
        secwin   = mainwin.secwin
        subtype  = control_panel.subtype.get()

        # Step 1. pass new parameters to the source_tree
        # get currect slection
        item = self.source_tree.selection()[0]
        values = self.source_tree.item(item, 'values')
        newvalues = list(values)

        # set period
        if np.isnan(period):
            newvalues[1] = ''
        else:
            newvalues[1] = '{:.6f}'.format(period)
        # set secondary phase
        if np.isnan(secphase):
            newvalues[2] = ''
        else:
            newvalues[2] = '{:6.4f}'.format(secphase)
        # set subtype
        newvalues[3] = subtype
        # set notes
        # read tag list from control panel
        tag_lst = [flag.upper()
                    for flag, var in control_panel.flags.items()
                        if var.get()
                    ]
        newvalues[4] = ', '.join(tag_lst)
        
        # display the new values in the source table
        self.source_tree.item(item, values=newvalues)

        # Step 2. pass the new parameters to he source table
        m = self.source_table['TIC']==tic
        idx = np.nonzero(m)[0]

        # median flux
        self.source_table['base_flux'][idx] = medflux
        # period
        if np.isnan(period):
            self.source_table['Period'][idx] = np.ma.masked
        else:
            self.source_table['Period'][idx] = period
        # T0
        if np.isnan(t0):
            self.source_table['T0'][idx] = np.ma.masked
        else:
            self.source_table['T0'][idx] = t0
        # T duration of primary and secondary eclipse
        if np.isnan(period):
            self.source_table['tdur_pri'][idx] = np.ma.masked
            self.source_table['tdur_sec'][idx] = np.ma.masked
        else:
            self.source_table['tdur_pri'][idx] = 2 * priwin * period
            self.source_table['tdur_sec'][idx] = 2 * secwin * period
        # phase of secondary eclipse
        if np.isnan(secphase):
            self.source_table['phi_sec'][idx] = np.ma.masked
        else:
            self.source_table['phi_sec'][idx] = secphase

        # list of unmasked sectors
        sector_used = [s for s in mainwin.sector_lst if mainwin.sector_mask[s]]
        # first and last of used sectors
        s1 = min(sector_used)
        s2 = max(sector_used)
        self.source_table['sector1'][idx] = s1
        self.source_table['sector2'][idx] = s2
        self.source_table['nsector'][idx] = len(sector_used)

        # find total tpsan of used light curves
        t1 = mainwin.tspan_lst[s1][0]  # start time of first used sector
        t2 = mainwin.tspan_lst[s2][1]  # end time of last used sector
        self.source_table['tspan'][idx] = t2 - t1

        # find total tlength of used light curves
        tlength_total = sum([tlen for s, tlen in mainwin.tlength_lst.items()
                                if mainwin.sector_mask[s]
                            ])
        self.source_table['tlength'][idx] = tlength_total

        # count number of primary and secondary eclipse
        npri, nsec, tspan_pri, tspan_sec = mainwin.count_neclipse()
        self.source_table['npri'][idx] = npri
        self.source_table['nsec'][idx] = nsec
        self.source_table['tspan_pri'][idx] = tspan_pri
        self.source_table['tspan_sec'][idx] = tspan_sec


        # subtype
        self.source_table['subtype'][idx] = subtype
        # tags
        for flag, var in control_panel.flags.items():
            if var.get():
                self.source_table[flag][idx] = 1

        # activiate save button
        self.source_changed = True
        self.save_button['state'] = tk.NORMAL


        # save folded light curve

        # save foleded figures
        figname = 'fig-fold-{:012d}_s{:03d}_s{:03d}.png'.format(
                mainwin.tic, s1, s2)
        figfilename = mainwin.figure_path / figname
        mainwin.plot_frame.fig.savefig(figfilename)


    def save_source_table(self):

        mainwin  = self.master.master

        self.source_table['base_flux'].info.format='%13.7e'
        self.source_table['Period'].info.format='%.7f'
        self.source_table['T0'].info.format='%13.7f'
        self.source_table['tdur_pri'].info.format='%10.7f'
        self.source_table['tdur_sec'].info.format='%10.7f'
        self.source_table['phi_sec'].info.format='%9.7f'
        self.source_table['tspan'].info.format='%9.4f'
        self.source_table['tlength'].info.format='%9.4f'
        self.source_table['tspan_pri'].info.format='%9.4f'
        self.source_table['tspan_sec'].info.format='%9.4f'
        self.source_table['subtype'].info.format='%-8s'

        self.source_table.write(self.source_filename,
                format='ascii.fixed_width_two_line', overwrite=True)

        # deactiviate save button
        self.source_changed = False
        self.save_button['state'] = tk.DISABLED


class SectorFrame(tk.Frame):
    def __init__(self, master, width, height):
        self.master = master
        tk.Frame.__init__(self, master, width=width, height=height)

        self.sector_tree = ttk.Treeview(self,
                columns = ('checked', 'Sector', 'MedFlux', 'tspan', 'tlength', '_'),
                show    = 'headings',
                style   = 'Treeview',
                height  = 10,
                selectmode = 'browse',
                )

        self.sector_tree.bind('<Button-1>', self.on_click)

        self.scrollbar = tk.Scrollbar(self,
                orient = tk.VERTICAL,
                width = 10,
                )
        self.sector_tree.column('checked', width=20, anchor='center', stretch=False)
        self.sector_tree.column('Sector', width=60, anchor='e')
        self.sector_tree.column('MedFlux', width=120, anchor='e')
        self.sector_tree.column('tspan', width=100, anchor='e')
        self.sector_tree.column('tlength', width=100, anchor='e')
        self.sector_tree.column('_', width=width-420, anchor='e')

        self.sector_tree.heading('checked', text='')
        self.sector_tree.heading('Sector', text='Sector')
        self.sector_tree.heading('MedFlux', text='Median Flux')
        self.sector_tree.heading('tspan', text='t_span (d)')
        self.sector_tree.heading('tlength', text='t_length (d)')
        self.sector_tree.config(yscrollcommand = self.scrollbar.set)

        style = ttk.Style()
        style.configure('Treeview', rowheight=30)

        self.scrollbar.config(command=self.sector_tree.yview)

        self.sector_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        self.pack()

    def load_sectors(self, sector_lst):

        # clear existing table
        for item in self.sector_tree.get_children():
            self.sector_tree.delete(item)

        for i, sector in enumerate(sector_lst):
            # find median flux of this sector
            medflux = self.master.master.medflux_lst[sector]
            t1, t2 = self.master.master.tspan_lst[sector]
            tspan   = t2 - t1
            tlength = self.master.master.tlength_lst[sector]
            item = (u'\u2611', sector,
                    '{:10.4e}'.format(medflux),
                    '{:7.4f}'.format(tspan),
                    '{:7.4f}'.format(tlength),
                    )
            iid = self.sector_tree.insert('', tk.END,
                    values=item, tags='checked')

    def on_click(self, event):
        iid = self.sector_tree.identify_row(event.y)
        column = self.sector_tree.identify_column(event.x)

        if column == '#1':
            tags = self.sector_tree.item(iid, 'tags')
            values = self.sector_tree.item(iid, 'values')
            sector = int(values[1])
            if tags[0]=='checked':
                if sum(list(self.master.master.sector_mask.values()))>1:
                    newvalues = list(values)
                    newvalues[0] = u'\u2610'
                    self.master.master.sector_mask[sector] = False
                    self.sector_tree.item(iid, values=newvalues, tags='unchecked')
            elif tags[0]=='unchecked':
                newvalues = list(values)
                newvalues[0] = u'\u2611'
                self.master.master.sector_mask[sector] = True
                self.sector_tree.item(iid, values=newvalues, tags='checked')
            else:
                pass
            self.master.master.plot()
            self.master.master.plot_frame.canvas.draw()

def smooth_trend(t_lst, f_lst, win):

    smooth_x = []
    smooth_y = []
    for t1 in np.arange(t_lst[0], t_lst[-1], win):
        t2 = t1 + win
        m = (t_lst >= t1) & (t_lst <= t2)
        if m.sum() <= 3:
            continue
        smooth_x.append(np.mean(t_lst[m]))
        smooth_y.append(np.mean(f_lst[m]))
    return np.array(smooth_x), np.array(smooth_y)

def launch(source_filename, datapool, figure_path=None, folded_path=None):
    master = tk.Tk()

    ### list fonts
    #import tkinter.font as tkFont
    #fonts = tkFont.families(master)
    #print(fonts)

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

    ratio = 2.5
    if window_width > window_height*ratio:
        window_width = window_height*ratio
    else:
        window_height = window_width//ratio

    window_width = int(round(window_width))
    window_height = int(round(window_height))

    x = int((screen_width-window_width)/2.)
    y = int((screen_height-window_height)/2.)

    master.geometry('{}x{}+{}+{}'.format(window_width, window_height, x, y))

    # prepare figure path
    if figure_path is None:
        figure_path = Path('./figures').resolve()
    else:
        figure_path = Path(figure_path).resolve()
    figure_path.mkdir(parents=True, exist_ok=True)

    # prepare folded light curve path
    if folded_path is None:
        folded_path = Path('./folded').resolve()
    else:
        folded_path = Path(folded_path).resolve()
    folded_path.mkdir(parents=True, exist_ok=True)

    # launch MainWindow
    mainwindow = MainWindow(master,
                            width  = window_width,
                            height = window_height,
                            source_filename = source_filename,
                            datapool = datapool,
                            figure_path = figure_path,
                            folded_path = folded_path,
                            )
    master.mainloop()


def get_overlap(d, R1, R2):
    A = np.zeros_like(d)
    # no eclipse
    mask0 = d >= (R1 + R2)
    A[mask0] = 0.0
    # total eclipse
    mask1 = d <= abs(R1 - R2)
    A[mask1] = np.pi * min(R1, R2)**2
    # partial eclipse
    mask2 = (~mask0) & (~mask1)
    d2 = d[mask2]
    term1 = R1**2 * np.arccos((d2**2 + R1**2 - R2**2) / (2 * d2 * R1))
    term2 = R2**2 * np.arccos((d2**2 + R2**2 - R1**2) / (2 * d2 * R2))
    term3 = 0.5 * np.sqrt(
        (-d2 + R1 + R2) * ( d2 + R1 - R2) * ( d2 - R1 + R2) * ( d2 + R1 + R2)
        )
    A[mask2] = term1 + term2 - term3
    return A

def ecl_errfunc_kopal(p, phase_lst, flux_lst):
    return flux_lst - get_ecl_kopal(phase_lst, p[0], p[1], p[2], p[3],
            p[4], p[5])

def ecl_errfunc_trapz(p, phase_lst, flux_lst):
    return flux_lst - get_ecl_trapz(phase_lst, p[0], p[1], p[2], p[3], p[4])

def get_ecl_kopal(phase_lst, F0, phase0, R1, R2, inc, u):
    """
    """

    phi_lst = 2 * np.pi * (phase_lst - phase0)
    d = np.sqrt(np.sin(phi_lst)**2 + (np.cos(inc) * np.cos(phi_lst))**2)

    # uniform brightness
    A = get_overlap(d, R1, R2)
    delta0 = A / (np.pi * R1**2)
    # first approximation of linear limb darkening
    delta1 = delta0 * (1 - 0.5 * delta0)
    # Kopal flux
    deltaF = (1 - u) * delta0 + u * delta1
    return F0*(1.0 - deltaF)

def get_kopal_phase14(r1, r2, inc):
    return np.arcsin(np.sqrt((r1 + r2)**2 - np.cos(inc)**2)/np.sin(inc))/np.pi

def get_ecl_trapz(phase_lst, F0, phase0, depth, ph14, ph23):
    ph12 = (ph14 - ph23)/2
    if ph12 <= 0:
        m1 = np.abs(phase_lst) > ph14/2
        flux = np.ones_like(phase_lst)
        flux[~m1] = 1 - depth
        return F0 * flux
    else:
        flux = np.ones_like(phase_lst)
        m1 = np.abs(phase_lst) < ph23/2
        flux[m1] = 1 - depth
        m2 = ( ~m1 ) * (np.abs(phase_lst) < ph14/2)
        k = depth/ph12
        b = 1 - depth*ph14/(ph14-ph23)
        flux[m2] =  k * np.abs(phase_lst[m2]) + b
        return F0 * flux
