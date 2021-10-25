# -*- coding: utf-8 -*-
"""
Tests for functionalities in ogr_util.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import ogr_util
import test_helper

def basetest_st_area():
    
    # try st_areag
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    sqlite_stmt = 'SELECT round(ST_area(geom), 2) as area FROM "parcels"'
    result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt)
    assert result_gdf is not None
    assert result_gdf['area'][0] == 146.8

    # Try st_makevalid 
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    sqlite_stmt = 'SELECT st_makevalid(geom) as geom FROM "parcels"'
    result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt)
    assert result_gdf['geometry'][0] is not None
    
    # Try st_isvalid 
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    sqlite_stmt = 'SELECT st_isvalid(geom) as geom FROM "parcels"'
    result_gdf = ogr_util._execute_sql(input_path, sqlite_stmt)
    assert result_gdf['geom'][0] is not None

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Test functionsv to run...
    #test_get_gdal_to_use()
    basetest_st_area()
