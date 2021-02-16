# -*- coding: utf-8 -*-
"""
Tests for functionalities in vector_util.
"""

from pathlib import Path
import sys

import shapely.geometry as sh_geom

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.util import vector_util

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

if __name__ == '__main__':
    import os
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "test_vector_util_geodataframe"
    os.makedirs(tmpdir, exist_ok=True)
