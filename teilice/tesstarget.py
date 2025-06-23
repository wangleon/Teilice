import os
import re
import math

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.mast import Tesscut
from astropy.units import UnitsWarning
import warnings
warnings.filterwarnings('ignore', category=UnitsWarning, append=True)

from .common import CACHE_PATH, TESSCUT_PATH, NEARBY_PATH
from .tessdata import read_lc, read_tp
from .utils import get_sectorinfo, download_lc, download_tp

def get_sourceinfo(tic):
    """Get basic information of input TIC target.

    Args:
        tic (int): TIC number.

    """
    catid = 'IV/39/tic82'
    tablelist = Vizier(catalog=catid, columns=['**'],
                column_filters={'TIC': '={}'.format(tic)}
                ).query_constraints()
    tictable = tablelist[catid]
    ticrow = tictable[0]

    gaia2 = ticrow['GAIA']
    if gaia2 is not np.ma.masked:
        catid = 'I/345/gaia2'
        tablelist = Vizier(catalog=catid, columns=['**'],
                    column_filters={'Source': '={}'.format(gaia2)}
                    ).query_constraints()
        gaiatable = tablelist[catid]
        gaiarow  = gaiatable[0]
    else:
        gaiarow = None
    return {
            'ticrow': ticrow,
            'gaiarow': gaiarow,
            'ra':   ticrow['RAJ2000'],
            'dec':  ticrow['DEJ2000'],
            'Tmag': ticrow['Tmag'],
            }

def query_nearbystars(coord, r):
    """Query the nearby TIC objects around a given coordinate.

    Args:
        coord (astropy.coordinates.SkyCoord): Astropy object of sky coordinate.
        r (float): radius in unti of arcsec.

    Returns:
        astropy.table.Table

    """
    #catid = 'IV/38/tic'
    catid = 'IV/39/tic82'
    viz = Vizier(catalog=catid, columns=['**', '+_r'])
    viz.ROW_LIMIT=-1
    tablelist = viz.query_region(coord, radius=r*u.arcsec,
                #column_filters={'Tmag': '<{}'.format(limit_mag)}
                )
    tictable = tablelist[catid]
    return tictable
    #filename = 'tic_nearby_{:011d}.vot'.format(tic)
    #tictable.write(filename, format='votable', overwrite=True)

class TessTarget(object):

    def __init__(self, tic):

        # check existence of cache folders
        #if not os.path.exists(CACHE_PATH):
        #    os.mkdir(CACHE_PATH)

        # get star info
        self.tic = tic

        # get nearby stars
        #self.get_nearbystars()

    def __getattr__(self, item):
        if item in ['ra', 'dec', 'tmag', 'coord', 'ticrow', 'gaiarow']:
            info = get_sourceinfo(self.tic)
            self.ra   = info['ra']
            self.dec  = info['dec']
            self.tmag = info['Tmag']
            self.coord = SkyCoord(self.ra, self.dec, unit='deg')
            self.ticrow = info['ticrow']
            self.gaiarow = info['gaiarow']
            return getattr(self, item)



    def get_nearbystars(self, r=250, cache_path=NEARBY_PATH):
        """Query nearby stars within given radius.

        Args:
            r (int): radius of searching region.
        """
        r = math.ceil(r)
        found_cache = False

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        # check if the target has already in the cache
        for fname in os.listdir(cache_path):
            mobj = re.match('tic_nearby_(\d+)_r(\d+)s\.vot', fname)
            if mobj:
                _tic = int(mobj.group(1))
                _r   = int(mobj.group(2))
                if _tic==self.tic and _r==r:
                    found_cache = True
                    break
        if found_cache:
            # nearby stars already in cache
            filename = os.path.join(NEARBY_PATH, fname)
            tictable = Table.read(filename)
        else:
            # nearby stars not in cache
            tictable = query_nearbystars(self.coord, r)
            fname = 'tic_nearby_{:011d}_r{:d}s.vot'.format(self.tic, r)
            filename = os.path.join(NEARBY_PATH, fname)
            tictable.write(filename, format='votable', overwrite=True)

        self.tictable = tictable

    def get_lc_sectors(self):
        """
        """
        filename = os.path.join(CACHE_PATH, 'tess_target_lc.dat')
        found = False
        file1 = open(filename)
        for row in file1:
            col = row.split(':')
            if int(col[0])==self.tic:
                sector_lst = [int(s) for s in col[1].split(',')]
                found = True
                break
        file1.close()
        if found:
            return sector_lst
        else:
            return []


    def get_sectors(self):
        """Get the observed sectors of target.
        """
        sector_table = Tesscut.get_sectors(coordinates=self.coord)
        return list(sector_table['sector'])


    def get_lc_filename(self, sector, datapool=None):
        timestamp, orbit = get_sectorinfo(sector)
        lcfile = 'tess{:013d}-s{:04d}-{:016d}-{:04d}-s_lc.fits'.format(
                    timestamp, sector, self.tic, orbit)
        path = os.path.join(datapool, 'lc', 's{:03d}'.format(sector))

        lcfilename = os.path.join(path, lcfile)
        return lcfilename

    def get_tp_filename(self, sector, datapool=None):
        
        timestamp, orbit = get_sectorinfo(sector)
        tpfile = 'tess{:013d}-s{:04d}-{:016d}-{:04d}-s_tp.fits'.format(
                    timestamp, sector, self.tic, orbit)
        path = os.path.join(datapool, 'tp', 's{:03d}'.format(sector))

        tpfilename = os.path.join(path, tpfile)
        return tpfilename


    def get_lc(self, sector, auto_download=True, datapool=None):
        # get LC filename
        lcfilename = self.get_lc_filename(sector, datapool=datapool)

        if not os.path.exists(lcfilename) and auto_download:
            download_lc(self.tic, sector, datapool)

        lcdata = read_lc(lcfilename)
        return lcdata
            
    def get_tp(self, sector, auto_download=True, datapool=None):

        # get TP filename
        tpfilename = self.get_tp_filename(sector, datapool=datapool)

        if not os.path.exists(tpfilename) and auto_download:
            download_tp(self.tic, sector, datapool)

        tpdata = read_tp(tpfilename)
        return tpdata

    def get_tesscutfile(self, sector, xsize, ysize, tesscut_path):
        pattern = 'tess\-s(\d{4})\-(\d)\-(\d)_(\d+\.\d+)_(\-?\d+\.\d+)_(\d+)x(\d+)_astrocut\.fits'
        if tesscut_path is None:
            tesscut_path = TESSCUT_PATH

        if not os.path.exists(tesscut_path):
            os.mkdir(tesscut_path)

        for fname in os.listdir(tesscut_path):
            mobj = re.match(pattern, fname)
            if mobj:
                _sector = int(mobj.group(1))
                if _sector != sector:
                    continue
                camera = int(mobj.group(2))
                ccd    = int(mobj.group(3))
                _ra    = float(mobj.group(4))
                _dec   = float(mobj.group(5))
                _xsize = int(mobj.group(6))
                _ysize = int(mobj.group(7))
                _coord = SkyCoord(_ra, _dec, unit='deg')
                _sep = self.coord.separation(_coord)
                _r = _sep.deg*3600  # convert separation to arcsec
                if _r < 1.0 and _xsize==xsize and _ysize==ysize:
                    filename = os.path.join(tesscut_path, fname)
                    return (filename, camera, ccd)

        return (None, None, None)

    def download_tesscut(self, sector, xsize, ysize, tesscut_path=None, method='wget'):
        """Download the tesscut images in a given sector and x/y sizes.
        
        Args:
            sector (int): Tess sector number.
            xsize (int): Number of pixels in X direction.
            ysize (int): Number of pixels in Y direction.
            tesscut_path (str): Path to save the TessCut files.
            method (str): Method of file downloading. Options include 'wget',
                and 'tesscut'.
        """
        if tesscut_path is None:
            tesscut_path = TESSCUT_PATH

        if method == 'wget':
            website = 'https://mast.stsci.edu/tesscut'
            url = '{}/api/v0.1/astrocut?ra={}&dec={}&y={}&x={}'.format(
                    website, self.ra, self.dec, ysize, xsize)
            outfile = 'tesscut_tic{:011d}_{}x{}.zip'.format(
                    self.tic, xsize, ysize)
            outfilename = os.path.join(tesscut_path, outfile)
            command = 'wget -O {} --content-disposition "{}"'.format(outfilename, url)
            os.system(command)
            command = 'unzip {} -d {}'.format(outfilename, tesscut_path)
            os.system(command)
            os.remove(outfilename)
        elif method == 'tesscut':
            Tesscut.download_cutouts(coordinates=self.coord, sector=sector,
                    size=(ysize, xsize), path=tesscut_path, inflate=True)
        else:
            raise ValueError
