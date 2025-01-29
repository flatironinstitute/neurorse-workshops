#!/usr/bin/env python3

import os.path as op
import shutil
from glob import glob

for f in glob(op.join('_site', 'browse_pages', '**', 'index.html'), recursive=True):
    print(f"{f} -> {f.replace(op.join('_site', 'browse_pages', ''), '')}")
    shutil.move(f, f.replace(op.join('_site', 'browse_pages', ''), ''))
