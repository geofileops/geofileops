# -*- coding: utf-8 -*-
"""
Tests for functionalities in vector_util.
"""

from pathlib import Path
import sys

import geopandas as gpd
import shapely.geometry as sh_geom

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
            crs='epsg:31370')
    assert len(grid_gdf) == 96

def test_split_tiles():
    input_tiles_path = get_testdata_dir() / 'BEFL_kbl.gpkg'
    input_tiles = geofile.read_file(input_tiles_path)
    nb_tiles_wanted = len(input_tiles) * 8
    result = vector_util.split_tiles(
            input_tiles=input_tiles,
            nb_tiles_wanted=nb_tiles_wanted)

    #geofile.to_file(result, r"C:\temp\BEFL_kbl_split.gpkg")

    assert len(result) == len(input_tiles) * 8

def test_simplify_ext(tmpdir):
    
    #### Test if keep_points_on works properly ####
    
    ## First init some stuff ##
    # Read the test data
    input_path = get_testdata_dir() / 'simplify_onborder_testcase.gpkg'
    input_gdf = geofile.read_file(input_path)

    # Create geometry where we want the points kept
    grid_gdf = vector_util.create_grid(
            total_bounds=(210431.875-1000, 176640.125-1000, 210431.875+1000, 176640.125+1000),
            nb_columns=2,
            nb_rows=2,
            crs='epsg:31370')
    geofile.to_file(grid_gdf, tmpdir / "grid.gpkg")
    grid_coords = [tile.exterior.coords for tile in grid_gdf['geometry']]
    grid_lines_geom = sh_geom.MultiLineString(grid_coords)
    
    ## Test ramer–douglas–peucker ##
    # Without keep_points_on, the following point that is on the test data + 
    # on the grid is removed by ramer–douglas–peucker with tolerance=1 
    point_on_input_and_border = sh_geom.Point(210431.875, 176599.375)
    tolerance_rdp = 1

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(input_gdf[input_gdf.intersects(point_on_input_and_border)])
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes 
    simplified_gdf = input_gdf.geometry.simplify(tolerance_rdp)
    print(len(simplified_gdf))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_rdp{tolerance_rdp}.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) != nb_intersects_with_input
    
    # With keep_points_on specified, the number of intersections stays the same 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(
                    geom, algorythm='ramer-douglas-peucker', 
                    tolerance=tolerance_rdp, keep_points_on=grid_lines_geom))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_rdp{tolerance_rdp}_keep_points_on.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) == nb_intersects_with_input
    
    ## Test visvalingam-whyatt ##
    # Without keep_points_on, the following point that is on the test data + 
    # on the grid is removed by ramer-douglas-peucker with tolerance=1 
    point_on_input_and_border = sh_geom.Point(210430.125, 176640.125)
    tolerance_vw = 16*0.25*0.25   # 1m²

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(input_gdf[input_gdf.intersects(point_on_input_and_border)])
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(geom, 'visvalingam-whyatt', tolerance=tolerance_vw))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_vw{tolerance_vw}.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) != nb_intersects_with_input
    
    # With keep_points_on specified, the number of intersections stays the same 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(geom, 'visvalingam-whyatt', tolerance=tolerance_vw, keep_points_on=grid_lines_geom))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_vw{tolerance_vw}_keep_points_on.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) == nb_intersects_with_input
    
if __name__ == '__main__':
    import os
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "test_vector_util"
    os.makedirs(tmpdir, exist_ok=True)
    #test_create_grid2()
    #test_split_tiles()
    test_simplify_ext(tmpdir)
