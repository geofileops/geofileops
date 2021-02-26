# -*- coding: utf-8 -*-
"""
Tests for functionalities in geoseries_util.
"""

from pathlib import Path
import sys

import geopandas as gpd

from .test_helper import TestData 
from .test_helper import get_testdata_dir 
# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.util import geometry_util
from geofileops.util import geoseries_util

def test_get_geometrytypes():
    test_gdf = gpd.GeoDataFrame(geometry=[
            TestData.point, TestData.multipoint, TestData.polygon, TestData.polygon2, TestData.multipolygon, TestData.geometrycollection])
    test_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_geometrytypes) == 5

def test_harmonize_geometrytypes():
    # Test for gdf with point + multipoint -> all multipoint
    test_gdf = gpd.GeoDataFrame(geometry=[
            TestData.point, TestData.multipoint])
    test_gdf_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_gdf_geometrytypes) == 2
    test_harmonized_gdf = geoseries_util.harmonize_geometrytypes(test_gdf.geometry)
    test_harmonized_geometrytypes = geoseries_util.get_geometrytypes(test_harmonized_gdf)
    assert len(test_harmonized_geometrytypes) == 1
    assert test_harmonized_geometrytypes[0] == geofile.GeometryType.MULTIPOINT

    # Test for gdf with linestring + multilinestring -> all multilinestring
    test_gdf = gpd.GeoDataFrame(geometry=[
            TestData.linestring, TestData.multilinestring])
    test_gdf_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_gdf_geometrytypes) == 2
    test_harmonized_gdf = geoseries_util.harmonize_geometrytypes(test_gdf.geometry)
    test_harmonized_geometrytypes = geoseries_util.get_geometrytypes(test_harmonized_gdf)
    assert len(test_harmonized_geometrytypes) == 1
    assert test_harmonized_geometrytypes[0] == geofile.GeometryType.MULTILINESTRING

    # Test for gdf with linestring + multilinestring -> all multilinestring
    test_gdf = gpd.GeoDataFrame(geometry=[
            TestData.polygon, TestData.multipolygon])
    test_gdf_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_gdf_geometrytypes) == 2
    test_harmonized_gdf = geoseries_util.harmonize_geometrytypes(test_gdf.geometry)
    test_harmonized_geometrytypes = geoseries_util.get_geometrytypes(test_harmonized_gdf)
    assert len(test_harmonized_geometrytypes) == 1
    assert test_harmonized_geometrytypes[0] == geofile.GeometryType.MULTIPOLYGON

    # Test for gdf with all types of geometrytypes -> no harmonization possible
    test_gdf = gpd.GeoDataFrame(geometry=[
            TestData.point, TestData.multipoint, TestData.polygon, TestData.polygon2, TestData.multipolygon, TestData.geometrycollection])
    test_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_geometrytypes) == 5
    test_harmonized_gdf = geoseries_util.harmonize_geometrytypes(test_gdf.geometry)
    test_harmonized_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_harmonized_geometrytypes) == 5
    
if __name__ == '__main__':
    import os
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "test_vector_util_geodataframe"
    tmpdir.mkdir(parents=True, exist_ok=True)
