import os
import re
import sys
import time
import astropy.units as u
from astroquery.vizier import Vizier
import urllib.request
from . import common as cm

def get_tictable(coord, r=250):
    #catid = 'IV/38/tic'
    catid = 'IV/39/tic82'
    viz = Vizier(catalog=catid, columns=['**', '+_r'])
    viz.ROW_LIMIT = -1
    tablelist = viz.query_region(coord, radius=r*u.arcsec)
    tictable = tablelist[catid]
    return tictable

def download_file(url, filename, show_progress=True):

    # if no dir, create new dirs
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    def get_human_readable_size(byte):
        unit = 'B'
        readable_size = byte
        if readable_size > 1024:
            readable_size /= 1024
            unit = 'kB'
        if readable_size > 1024:
            readable_size /= 1024
            unit = 'MB'
        if readable_size > 1024:
            readable_size /= 1024
            unit = 'GB'
        return readable_size, unit
    
    def get_human_readable_time(second):
        readable_time = second
        if second < 60:
            return '{:02d}s'.format(int(second))
    
        minute = int(second/60)
        second = second - minute*60
        if minute < 60:
            return '{:d}m{:02d}s'.format(minute, int(second))
    
        hour = int(minute/60)
        minute = minute - hour*60
        if hour < 24:
            return '{:d}h{:02d}m'.format(hour, int(minute))
    
        day = int(hour/24)
        hour = hour - day*24
        return '{:d}d{:02d}h'.format(day, int(hour))
    
    def callback(block_num, block_size, total_size):
        t1 = time.time()
        # downloaded size in byte
        down_size = block_num * block_size
        speed = (down_size - param[1])/(t1-param[0])
        if speed > 0:
            # ETA in seconds
            eta = (total_size - down_size)/speed
            eta = get_human_readable_time(eta)
        else:
            eta = '--'
        ratio = min(down_size/total_size, 1.0)
        percent = min(ratio*100., 100)
    
        disp_speed, unit_speed = get_human_readable_size(speed)
        disp_size,  unit_size  = get_human_readable_size(total_size)
    
        # get the width of the terminal
        term_size = os.get_terminal_size()
    
        str1 = 'Downloading {}'.format(os.path.basename(filename))
        str2 = '{:6.2f}% of {:6.1f} {}'.format(percent, disp_size, unit_size)
        str3 = '{:6.1f} {}/s'.format(disp_speed, unit_speed)
        str4 = 'ETA: {}'.format(eta)
    
        n = term_size.columns-len(str1)-len(str2)-len(str3)-len(str4)-20
        progbar = '>'*int(ratio*n)
        progbar = progbar.ljust(n, '-')
    
        msg = '\r {} |{}| {} {} {}'.format(str1, progbar, str2, str3, str4)
        sys.stdout.write(msg)
        sys.stdout.flush()

    param = [time.time(), 0]
    # download
    if show_progress:
        urllib.request.urlretrieve(url, filename, callback)
        # use light green color
        print('\033[92m Completed\033[0m')
    else:
        urllib.request.urlretrieve(url, filename)


def download_sector_script(filetype, sector, show_progress=True):
    path = 'https://archive.stsci.edu/missions/tess/download_scripts/sector/'
    fname = 'tesscurl_sector_{}_{}.sh'.format(sector, filetype)
    url = os.path.join(path, fname)
    filename = os.path.join(cm.DOWNSCRIPT_PATH, 'sector', fname)
    download_file(url, filename, show_progress=show_progress)

def download_all_sector_scripts():
    site = 'https://archive.stsci.edu'
    url0 = site + '/tess/bulk_downloads/bulk_downloads_ffi-tp-lc-dv.html'
    response = urllib.request.urlopen(url0)
    content = response.read().decode('utf-8')
    rowlist = content.split('\n')
    pattern = '\s*<td [\s\S]*><a href="(\S+)">(tesscurl_sector_\d+_[a-z\-]+\.sh)<\/a><\/td>'
    for row in rowlist:
        mobj = re.match(pattern, row)
        if mobj:
            url = site + mobj.group(1)
            fname = mobj.group(2)
            filename = os.path.join(cm.DOWNSCRIPT_PATH, 'sector', fname)
            if not os.path.exists(filename):
                download_file(url, filename, show_progress=True)
