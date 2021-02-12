# -*- coding: utf-8 -*-
"""
Tests for functionalities in ogr_util.
"""

from pathlib import Path
import pprint
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import ogr_util
from tests import test_helper

def test_get_gdal_to_use():

    # If GDAL_BIN not set
    try:
        ogr_util.get_gdal_to_use('ST_area()')
        test_ok = True
    except ogr_util.SQLNotSupportedException:
        test_ok = False
    assert test_ok is test_helper.is_gdal_ok('area', 'gdal_default')
    
    # If GDAL_BIN set
    with test_helper.GdalBin('gdal_bin'):
        try:
            ogr_util.get_gdal_to_use('ST_area()')
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
        assert test_ok is test_helper.is_gdal_ok('area', 'gdal_bin')

def test_gis_operations():

    # Depends on the spatialite version
    gdal_installation = 'gdal_default'
    install_info = ogr_util.get_gdal_install_info(gdal_installation)
    if install_info['spatialite_version()'] >= '5.0.0':
        if install_info['rttopo_version()'] is None:
            basetest_st_area(gdal_installation, sql_dialect='INDIRECT_SQLITE')
        else:
            basetest_st_area(gdal_installation, sql_dialect='SQLITE')
    elif install_info['spatialite_version()'] >= '4.3.0':
        if install_info['lwgeom_version()'] is None:
            basetest_st_area(gdal_installation, sql_dialect='INDIRECT_SQLITE')
        else:
            basetest_st_area(gdal_installation, sql_dialect='SQLITE')

def basetest_st_area(
        gdal_installation: str, 
        sql_dialect: str):
    
    # try st_areag
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    sqlite_stmt = 'SELECT round(ST_area(geom), 2) as area FROM "parcels"'
    test_ok = False
    ok_expected = test_helper.is_gdal_ok('', gdal_installation)
    result_gdf = None
    try:
        result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt, gdal_installation, sql_dialect)
        test_ok = True
    except ogr_util.SQLNotSupportedException:
        test_ok = False
    assert test_ok is ok_expected, f"Result: {test_ok}, expected: {ok_expected}, with stmt <{sqlite_stmt}> and gdal_installation: {gdal_installation}, install_info: {ogr_util.get_gdal_install_info(gdal_installation)}"  
    assert result_gdf is not None
    assert result_gdf['area'][0] == 146.8

    # Try st_makevalid 
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    sqlite_stmt = 'SELECT st_makevalid(geom) as geom FROM "parcels"'
    test_ok = False
    ok_expected = test_helper.is_gdal_ok('', gdal_installation)
    try:
        result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt, gdal_installation, sql_dialect)
        if result_gdf['geometry'][0] is not None:
            test_ok = True
    except ogr_util.SQLNotSupportedException:
        test_ok = False
    assert test_ok is ok_expected, f"Test to run test <{sqlite_stmt}> failed for gdal_installation: {gdal_installation}, install_info: {ogr_util.get_gdal_install_info(gdal_installation)}"  
    
    # Try st_isvalid 
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    sqlite_stmt = 'SELECT st_isvalid(geom) as geom FROM "parcels"'
    test_ok = False
    ok_expected = test_helper.is_gdal_ok('', gdal_installation)
    try:
        result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt, gdal_installation, sql_dialect)
        if result_gdf['geom'][0] is not None:
            test_ok = True
    except ogr_util.SQLNotSupportedException:
        test_ok = False
    assert test_ok is ok_expected, f"Test to run test <{sqlite_stmt}> failed for gdal_installation: {gdal_installation}, install_info: {ogr_util.get_gdal_install_info(gdal_installation)}"  

if __name__ == '__main__':
    import tempfile
    tmpdir = tempfile.gettempdir()

    # First print out some version info of the spatialite used
    print(f"gdal_default: {pprint.pformat(ogr_util.get_gdal_install_info('gdal_default'))}")
    with test_helper.GdalBin('gdal_bin'):
        print(f"gdal_bin: {pprint.pformat(ogr_util.get_gdal_install_info('gdal_bin'))}")
    
    # Test functionsv to run...
    #test_get_gdal_to_use()
    test_gis_operations()
