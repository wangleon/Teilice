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


def get_sectorinfo(sector, filetype='lc'):

    script_fname = 'tesscurl_sector_{}_{}.sh'.format(sector, filetype)
    script_file = os.path.join(cm.DOWNSCRIPT_PATH, 'sector', script_fname)

    ### open the download script file and find the timestamp and orbit number
    file1 = open(script_file)
    file1.readline()
    row = file1.readline()
    file1.close()

    # get timestamp and orbit of this sector
    mobj1 = re.match('[\s\S]+tess(\d+)\-s\d+\-\d+\-(\d+)\-\S+\.fits', row)
    timestamp = int(mobj1.group(1))
    orbit = int(mobj1.group(2))

    return timestamp, orbit



def download_data(tic, sector, filetype, datapool, show_progress=True):
    script_fname = 'tesscurl_sector_{}_{}.sh'.format(sector, filetype)
    script_file = os.path.join(cm.DOWNSCRIPT_PATH, 'sector', script_fname)

    ### open the download script file and find the timestamp and orbit number
    file1 = open(script_file)
    file1.readline()
    row = file1.readline()
    file1.close()

    # get timestamp and orbit of this sector
    mobj1 = re.match('[\s\S]+tess(\d+)\-s\d+\-\d+\-(\d+)\-\S+\.fits', row)
    timestamp = int(mobj1.group(1))
    orbit = int(mobj1.group(2))

    # get download URL
    mobj2 = re.match('[\s\S]+(https:\/\/\S+)tess\d+\-s\d+\-\d+\-\d+\S+', row)
    url = mobj2.group(1)

    fname = 'tess{:013d}-s{:04d}-{:016d}-{:04d}-s_{}.fits'.format(
            timestamp, sector, tic, orbit, filetype)

    full_url = url+fname

    # check if target path exists
    target_path = os.path.join(datapool, filetype, 's{:03d}'.format(sector))
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    filename = os.path.join(target_path, fname)

    # download the data
    download_file(full_url, filename, show_progress=show_progress)


def download_lc(tic, sector, datapool, show_progress=True):
    download_data(tic, sector, 'lc', datapool)

def download_tp(tic, sector, datapool, show_progress=True):
    download_data(tic, sector, 'tp', datapool)


def make_target_lst():
    tic_lst = {}

    path1 = os.path.join(cm.DOWNSCRIPT_PATH, 'sector')
    for fname in os.listdir(path1):
        mobj = re.match('tesscurl_sector_(\d+)_lc.sh', fname)
        if not mobj:
            continue
        sector = int(mobj.group(1))

        filename = os.path.join(path1, fname)
        file1 = open(filename)
        for row in file1:
            if len(row)==0 or row[0]=='#':
                continue
            mobj = re.match('[\s\S]+tess\d+\-s\d+\-(\d+)\-\d+\S+\.fits', row)
            if mobj:
                tic = int(mobj.group(1))
                if tic not in tic_lst:
                    tic_lst[tic] = []
                tic_lst[tic].append(sector)
        file1.close()

    filename = os.path.join(cm.CACHE_PATH, 'tess_target_lc.dat')
    outfile = open(filename, 'w')
    for tic, sector_lst in sorted(tic_lst.items()):
        outfile.write('{:11d}:'.format(tic))
        string_lst = [str(s) for s in sorted(sector_lst)]
        outfile.write(','.join(string_lst)+os.linesep)
    outfile.close()

    print('LC target list updated. N={}'.format(len(tic_lst)))
