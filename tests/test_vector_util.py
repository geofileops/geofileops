# -*- coding: utf-8 -*-
"""
Tests for functionalities in vector_util.
"""

from pathlib import Path
import shapely.geometry as sh_geom
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))
from geofileops import geofile
from geofileops.util import vector_util

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_create_grid2():
    # Test for small number of cells
    '''
    for i in range(1, 10):
        grid_gdf = vector_util.create_grid2(
                total_bounds=(40000.0, 160000.0, 45000.0, 210000.0), 
                nb_squarish_cells=i)
        assert len(grid_gdf) == i
    '''
    
    # Test for larger number of cells
    grid_gdf = vector_util.create_grid2(
            total_bounds=(40000.0, 160000.0, 45000.0, 210000.0), 
            nb_squarish_tiles=100,
            crs=None)
    assert len(grid_gdf) == 132

def test_split_tiles():
    input_tiles_path = get_testdata_dir() / 'BEFL_kbl.gpkg'
    input_tiles = geofile.read_file(input_tiles_path)
    nb_tiles_wanted = len(input_tiles) * 8
    result = vector_util.split_tiles(
            input_tiles=input_tiles,
            nb_tiles_wanted=nb_tiles_wanted)

    #geofile.to_file(result, r"C:\temp\BEFL_kbl_split.gpkg")

    assert len(result) == len(input_tiles) * 8

def test_simplify_ext():
    input_path = get_testdata_dir() / 'simplify_onborder_testcase.gpkg'
    input_gdf = geofile.read_file(input_path)

    # Create grid lines geometry 
    grid_gdf = vector_util.create_grid(
            total_bounds=(210431.875-1000, 176640.125-1000, 210431.875+1000, 176640.125+1000),
            nb_columns=2,
            nb_rows=2,
            crs='epsg:31370')
    grid_coords = [tile.exterior.coords for tile in grid_gdf['geometry']]
    grid_lines = sh_geom.MultiLineString(grid_coords)

    simplified_gdf = input_gdf.geometry.simplify(1)
    geofile.to_file(simplified_gdf, r"c:\temp\simplified.gpkg")

    simplified_dist_ext_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(geom, 'ramer–douglas–peucker', tolerance=tolerance, keep_points_on=grid_lines))
    geofile.to_file(simplified_dist_ext_gdf, r"c:\temp\simplified_dist0.25_ext.gpkg")

    tolerance = 16*0.25*0.25
    simplified_area_ext_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(geom, 'visvalingam-whyatt', tolerance=tolerance, keep_points_on=grid_lines))
    geofile.to_file(simplified_area_ext_gdf, f"c:/temp/simplified_area{tolerance}_ext.gpkg")

if __name__ == '__main__':
    #import tempfile
    #tmpdir = tempfile.gettempdir()
    #test_create_grid2()
    #test_split_tiles()
    test_simplify_ext()
