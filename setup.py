#!/usr/bin/env python3
from distutils.core import setup

setup(
    name         = 'teilice',
    version      = '0.3',
    description  = 'TESS Full Frame Image Photometry Package',
    author       = 'Liang Wang',
    author_email = 'wang.leon@gmail.com',
    license      = 'Apache-2.0',
    packages     = [
                    'teilice',
                    'teilice/tools',
                    ],

    )
