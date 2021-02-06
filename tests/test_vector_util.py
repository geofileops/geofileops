# -*- coding: utf-8 -*-
"""
Tests for functionalities in vector_util.
"""

from pathlib import Path
import sys

import geopandas as gpd
import shapely.geometry as sh_geom
from shapely.geometry import linestring

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

def test_numberpoints():
    # Test Point
    point = sh_geom.Point((0, 0))
    numberpoints = vector_util.numberpoints(point)
    numberpoints_geometrycollection = numberpoints
    assert numberpoints == 1

    # Test MultiPoint
    multipoint = sh_geom.MultiPoint([(0, 0), (10, 10), (20, 20)])
    numberpoints = vector_util.numberpoints(multipoint)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 3
    
    # Test LineString
    linestring = sh_geom.LineString([(0, 0), (10, 10), (20, 20)])
    numberpoints = vector_util.numberpoints(linestring)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 3
    
    # Test MultiLineString
    multilinestring = sh_geom.MultiLineString([[(0, 0), (10, 10), (20, 20)], [(100, 100), (110, 110), (120, 120)]])
    numberpoints = vector_util.numberpoints(multilinestring)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 6

    # Test Polygon
    poly = sh_geom.Polygon(shell=[(0, 0), (0, 10), (10, 10), (10, 0), (0,0)] , holes=[[(2,2), (2,8), (8,8), (8,2), (2,2)]])
    numberpoints = vector_util.numberpoints(poly)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 10

    # Test MultiPolygon
    poly2 = sh_geom.Polygon(shell=[(100, 100), (100, 110), (110, 110), (110, 100), (100,100)])
    multipoly = sh_geom.MultiPolygon([poly, poly2])
    numberpoints = vector_util.numberpoints(multipoly)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 15

    # Test GeometryCollection (as combination of all previous ones)
    geometrycollection = sh_geom.GeometryCollection([point, multipoint, linestring, multilinestring, poly, multipoly])
    numberpoints = vector_util.numberpoints(geometrycollection)
    assert numberpoints == numberpoints_geometrycollection

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
    geofile.copy(input_path, tmpdir / input_path.name)
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
    
    ## Test rdp (ramer–douglas–peucker) ##
    # Without keep_points_on, the following point that is on the test data + 
    # on the grid is removed by rdp 
    point_on_input_and_border = sh_geom.Point(210431.875, 176599.375)
    tolerance_rdp = 0.5

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(input_gdf[input_gdf.intersects(point_on_input_and_border)])
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(
                    geom, algorithm=vector_util.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER, 
                    tolerance=tolerance_rdp))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_rdp{tolerance_rdp}.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) != nb_intersects_with_input
    
    # With keep_points_on specified, the number of intersections stays the same 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(
                    geom, algorithm=vector_util.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER, 
                    tolerance=tolerance_rdp, keep_points_on=grid_lines_geom))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_rdp{tolerance_rdp}_keep_points_on.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) == nb_intersects_with_input
    
    ## Test vw (visvalingam-whyatt) ##
    # Without keep_points_on, the following point that is on the test data + 
    # on the grid is removed by vw 
    point_on_input_and_border = sh_geom.Point(210430.125, 176640.125)
    tolerance_vw = 16*0.25*0.25   # 1m²

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(input_gdf[input_gdf.intersects(point_on_input_and_border)])
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(
                    geom, algorithm=vector_util.SimplifyAlgorithm.VISVALINGAM_WHYATT, 
                    tolerance=tolerance_vw))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_vw{tolerance_vw}.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) != nb_intersects_with_input
    
    # With keep_points_on specified, the number of intersections stays the same 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(
                    geom, algorithm=vector_util.SimplifyAlgorithm.VISVALINGAM_WHYATT, 
                    tolerance=tolerance_vw, 
                    keep_points_on=grid_lines_geom))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_vw{tolerance_vw}_keep_points_on.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) == nb_intersects_with_input
    
    ## Test lang ##
    # Without keep_points_on, the following point that is on the test data + 
    # on the grid is removed by lang 
    point_on_input_and_border = sh_geom.Point(210431.875,176606.125)
    tolerance_lang = 0.25
    step_lang = 8

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(input_gdf[input_gdf.intersects(point_on_input_and_border)])
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(
                    geom, algorithm=vector_util.SimplifyAlgorithm.LANG, 
                    tolerance=tolerance_lang, lookahead=step_lang))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_lang;{tolerance_lang};{step_lang}.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) != nb_intersects_with_input
    
    # With keep_points_on specified, the number of intersections stays the same 
    simplified_gdf = input_gdf.geometry.apply(
            lambda geom: vector_util.simplify_ext(
                    geom, algorithm=vector_util.SimplifyAlgorithm.LANG, 
                    tolerance=tolerance_lang, lookahead=step_lang, 
                    keep_points_on=grid_lines_geom))
    geofile.to_file(simplified_gdf, tmpdir / f"simplified_lang;{tolerance_lang};{step_lang}_keep_points_on.gpkg")
    assert len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)]) == nb_intersects_with_input
    
if __name__ == '__main__':
    import os
    import tempfile
    tmpdir = Path(tempfile.gettempdir()) / "test_vector_util"
    os.makedirs(tmpdir, exist_ok=True)
    #test_create_grid2()
    test_numberpoints()
    #test_split_tiles()
    #test_simplify_ext(tmpdir)
