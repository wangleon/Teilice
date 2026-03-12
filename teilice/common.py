import os

CACHE_PATH = os.path.expanduser('~/.teilice')
TESSCUT_PATH = os.path.join(CACHE_PATH, 'tesscut')
NEARBY_PATH  = os.path.join(CACHE_PATH, 'nearby')
DOWNSCRIPT_PATH = os.path.join(CACHE_PATH, 'download_scripts')
LC_TARGETS_FILE = os.path.join(CACHE_PATH, 'tess_target_lc.dat')
FASTLC_TARGETS_FILE = os.path.join(CACHE_PATH, 'tess_target_fast-lc.dat')
