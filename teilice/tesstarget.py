import os
import re
import math
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.mast import Tesscut

from .common import CACHE_PATH, TESSCUT_PATH, NEARBY_PATH

def get_sourceinfo(tic):
    """Get basic information of input TIC target.

    Args:
        tic (int): TIC number.

    """
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
    """Query the nearby TIC objects around a given coordinate.

    Args:
        coord (astropy.coordinates.SkyCoord): Astropy object of sky coordinate.
        r (float): radius in unti of arcsec.

    Returns:
        astropy.table.Table

    """
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

class TessTarget(object):

    def __init__(self, tic):

        # check existence of cache folders
        if not os.path.exists(CACHE_PATH):
            os.mkdir(CACHE_PATH)

        # get star info
        self.tic = tic
        info = get_sourceinfo(tic)
        self.ra   = info['ra']
        self.dec  = info['dec']
        self.tmag = info['Tmag']
        self.coord = SkyCoord(self.ra, self.dec, unit='deg')

        # get nearby stars
        self.get_nearbystars()

    def get_nearbystars(self, r=250):
        """Query nearby stars within given radius.

        Args:
            r (int): radius of searching region.
        """
        r = math.ceil(r)
        found_cache = False

        if not os.path.exists(NEARBY_PATH):
            os.mkdir(NEARBY_PATH)

        # check if the target has already in the cache
        for fname in os.listdir(NEARBY_PATH):
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

    def get_sectors(self):
        """Get the observed sectors of target.
        """
        sector_table = Tesscut.get_sectors(coordinates=self.coord)
        return list(sector_table['sector'])

    def get_tesscutfile(self, sector, xsize, ysize):
        pattern = 'tess\-s(\d{4})\-(\d)\-(\d)_(\d+\.\d+)_(\-?\d+\.\d+)_(\d+)x(\d+)_astrocut\.fits'
        if not os.path.exists(TESSCUT_PATH):
            os.mkdir(TESSCUT_PATH)

        for fname in os.listdir(TESSCUT_PATH):
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
                    filename = os.path.join(TESSCUT_PATH, fname)
                    return (filename, camera, ccd)

        return (None, None, None)

    def download_tesscut(self, sector, xsize, ysize, method='wget'):
        """Download the tesscut images in a given sector and x/y sizes.
        
        Args:
            sector (int):
            xsize (int):
            ysize (int):
            method (str): Method of file downloading. Options include 'wget',
                and 'tesscut'.
        """
        if method == 'wget':
            website = 'https://mast.stsci.edu/tesscut'
            url = '{}/api/v0.1/astrocut?ra={}&dec={}&y={}&x={}'.format(
                    website, self.ra, self.dec, ysize, xsize)
            outfile = 'tesscut_tic{:011d}_{}x{}.zip'.format(
                    self.tic, xsize, ysize)
            outfilename = os.path.join(TESSCUT_PATH, outfile)
            command = 'wget -O {} --content-disposition "{}"'.format(outfilename, url)
            os.system(command)
            command = 'unzip {} -d {}'.format(outfilename, TESSCUT_PATH)
            os.system(command)
            os.remove(outfilename)
        elif method == 'tesscut':
            Tesscut.download_cutouts(coordinates=self.coord, sector=sector,
                    size=(ysize, xsize), path=TESSCUT_PATH, inflate=True)
        else:
            raise ValueError
