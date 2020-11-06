# -*- coding: utf-8 -*-
"""
Helper functions for all tests.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops.util import ogr_util

class GdalBin():
    def __init__(self, gdal_installation: str, gdal_bin_path: str = None):
        self.gdal_installation = gdal_installation
        if gdal_installation == 'gdal_bin':
            if gdal_bin_path is None:
                self.gdal_bin_path = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
            else:
                self.gdal_bin_path = gdal_bin_path

    def __enter__(self):
        if self.gdal_installation == 'gdal_bin':
            import os
            os.environ['GDAL_BIN'] = self.gdal_bin_path

    def __exit__(self, type, value, traceback):
        #Exception handling here
        import os
        if os.environ.get('GDAL_BIN') is not None:
            del os.environ['GDAL_BIN']

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def is_gdal_ok(operation: str, gdal_installation: str) -> bool:
    # Check if there are unsupported functions
    install_info = ogr_util.get_gdal_install_info(gdal_installation)
    if install_info['spatialite_version()'] >= '5.0.0':
        if install_info['rttopo_version()'] is None:
            if operation in ['makevalid', 'isvalid']:
                return False
            else:
                return True
        else:
            return True
    elif install_info['spatialite_version()'] >= '4.3.0':
        if install_info['lwgeom_version()'] is None:
            return False
        else:
            return True
    return False
