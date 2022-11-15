import astropy.units as u
from astroquery.vizier import Vizier

def get_tictable(coord, r=250):
    #catid = 'IV/38/tic'
    catid = 'IV/39/tic82'
    viz = Vizier(catalog=catid, columns=['**', '+_r'])
    viz.ROW_LIMIT = -1
    tablelist = viz.query_region(coord, radius=r*u.arcsec)
    tictable = tablelist[catid]
    return tictable

