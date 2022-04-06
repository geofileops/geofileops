# -*- coding: utf-8 -*-
"""
Tests for functionalities in ogr_util.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import ogr_util
from tests import test_helper

def test_get_drivers():

    drivers = ogr_util.get_drivers()
    assert len(drivers) > 0
    assert "GPKG" in drivers
    assert "ESRI Shapefile" in drivers 

def test_execute_st_area():
    
    # try st_area
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
