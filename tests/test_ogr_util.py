# -*- coding: utf-8 -*-
"""
Tests for functionalities in ogr_util.
"""

import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops.util import ogr_util
from geofileops import geofile
from tests import test_helper

def test_get_gdal_to_use():

    # On windows, the default gdal installation with conda doesn't work
    if os.name == 'nt': 
        # If GDAL_BIN not set, should be not OK
        try:
            ogr_util.get_gdal_to_use('ST_area()')
            test_ok = True
        except:
            test_ok = False
        assert test_ok is test_helper.is_gdal_ok('area', 'gdal_default'), "On windows, check is expected to be OK if GDAL_BIN is not set"

        # If GDAL_BIN set, it should be ok as well
        with test_helper.GdalBin('gdal_bin'):
            try:
                ogr_util.get_gdal_to_use('ST_area()')
                test_ok = True
            except:
                test_ok = False
            assert test_ok is test_helper.is_gdal_ok('area', 'gdal_bin'), "On windows, check is expected to be OK if GDAL_BIN is set (properly)"
        
    else:
        try:
            ogr_util.get_gdal_to_use('ST_area()')
            test_ok = True
        except:
            test_ok = False
        assert test_ok is True, "If not on windows, check is expected to be OK without setting GDAL_BIN"

def test_gis_operations():

    # Depends on the spatialite version
    gdal_installation = 'gdal_default'
    install_info = ogr_util.get_gdal_install_info(gdal_installation)
    if install_info['spatialite_version()'] >= '5.0.0':
        if install_info['rttopo_version()'] is None:
            basetest_st_area(gdal_installation)
        else:
            basetest_st_area(gdal_installation)
    elif install_info['spatialite_version()'] >= '4.3.0':
        if install_info['lwgeom_version()'] is None:
            basetest_st_area(gdal_installation)
        else:
            basetest_st_area(gdal_installation)

def basetest_st_area(gdal_installation: str):
    
    # try st_area
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    sqlite_stmt = 'SELECT round(ST_area(geom), 2) as area FROM "parcels"'
    test_ok = False
    try:
        result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt, gdal_installation)
        if result_gdf['area'][0] == 146.8:
            test_ok = True
    except:
        assert False == test_helper.is_gdal_ok('makevalid', gdal_installation)
        test_ok = True
    assert test_ok is True, f"Test to run test <{sqlite_stmt}> failed for gdal_installation: {gdal_installation}, install_info: {ogr_util.get_gdal_install_info(gdal_installation)}"  
    
    # Try st_makevalid 
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    sqlite_stmt = 'SELECT st_makevalid(geom) as geom FROM "parcels"'
    test_ok = False
    try:
        result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt, gdal_installation)
        if result_gdf['geom'][0] is not None:
            test_ok = True
    except:
        assert False == test_helper.is_gdal_ok('makevalid', gdal_installation)
        test_ok = True
    assert test_ok is True, f"Test to run test <{sqlite_stmt}> failed for gdal_installation: {gdal_installation}, install_info: {ogr_util.get_gdal_install_info(gdal_installation)}"  
    
    # Try st_isvalid 
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    sqlite_stmt = 'SELECT st_isvalid(geom) as geom FROM "parcels"'
    test_ok = False
    try:
        result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt, gdal_installation)
        if result_gdf['geom'][0] is not None:
            test_ok = True
    except:
        assert False == test_helper.is_gdal_ok('makevalid', gdal_installation)
        test_ok = True
    assert test_ok is True, f"Test to run test <{sqlite_stmt}> failed for gdal_installation: {gdal_installation}, install_info: {ogr_util.get_gdal_install_info(gdal_installation)}"  

if __name__ == '__main__':
    import tempfile
    tmpdir = tempfile.gettempdir()

    test_get_gdal_to_use()
