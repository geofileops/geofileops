# -*- coding: utf-8 -*-
"""
Helper functions for all tests.
"""

import logging
from pathlib import Path
import sys
import tempfile

import shapely.geometry as sh_geom

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class TestData:
    point = sh_geom.Point((0, 0))
    multipoint = sh_geom.MultiPoint([(0, 0), (10, 10), (20, 20)])
    linestring = sh_geom.LineString([(0, 0), (10, 10), (20, 20)])
    multilinestring = sh_geom.MultiLineString(
            [linestring.coords, [(100, 100), (110, 110), (120, 120)]])
    polygon = sh_geom.Polygon(
            shell=[(0, 0), (0, 10), (1, 10), (10, 10), (10, 0), (0,0)], 
            holes=[[(2,2), (2,8), (8,8), (8,2), (2,2)]])
    polygon2 = sh_geom.Polygon(shell=[(100, 100), (100, 110), (110, 110), (110, 100), (100,100)])
    polygon3 = sh_geom.Polygon(
            shell=[(20, 20), (20, 30), (21, 30), (30, 30), (30, 20), (20,20)], 
            holes=[[(22,22), (22,28), (28,28), (28,22), (22,22)]])
    multipolygon = sh_geom.MultiPolygon([polygon2, polygon3])
    geometrycollection = sh_geom.GeometryCollection([
            point, multipoint, linestring, multilinestring, polygon, multipolygon])

class TestFiles:
    testdata_dir = Path(__file__).resolve().parent / 'data'

    BEFL_kbl_gpkg = testdata_dir / 'BEFL_kbl.gpkg'

    linestrings_rows_of_trees_gpkg = testdata_dir / 'linestrings_rows_of_trees.gpkg'
    linestrings_watercourses_gpkg = testdata_dir / 'linestrings_watercourses.gpkg'

    polygons_no_rows_gpkg = testdata_dir / 'polygons_no_rows.gpkg'
    polygons_overlappingcircles_all_gpkg = testdata_dir / 'polygons_overlappingcircles_all.gpkg'
    polygons_overlappingcircles_one_gpkg = testdata_dir / 'polygons_overlappingcircles_one.gpkg'
    polygons_overlappingcircles_twothree_gpkg = testdata_dir / 'polygons_overlappingcircles_two+three.gpkg'
    polygons_parcels_gpkg = testdata_dir / 'polygons_parcels-2020.gpkg'
    polygons_parcels_shp = testdata_dir / 'polygons_parcels-2020.shp'
    polygons_invalid_geometries_gpkg = testdata_dir / 'polygons_invalid_geometries.gpkg'
    polygons_invalid_geometries_shp = testdata_dir / 'polygons_invalid_geometries.shp'
    polygons_simplify_onborder_testcase_gpkg = testdata_dir / 'polygons_simplify_onborder_testcase.gpkg'
    polygons_twolayers_gpkg = testdata_dir / 'polygons_twolayers.gpkg'
    polygons_zones_gpkg = testdata_dir / 'polygons_zones.gpkg'
    polygons_zones_shp = testdata_dir / 'polygons_zones.shp'
    
    points_gpkg = testdata_dir / 'points.gpkg'

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def create_tempdir(
        base_dirname: str,
        parent_dir: Path = None) -> Path:
    # Parent
    if parent_dir is None:
        parent_dir = Path(tempfile.gettempdir())

    for i in range(1, 999999):
        try:
            tempdir = parent_dir / f"{base_dirname}_{i:06d}"
            tempdir.mkdir(parents=True)
            return Path(tempdir)
        except FileExistsError:
            continue

    raise Exception(f"Wasn't able to create a temporary dir with basedir: {parent_dir / base_dirname}") 

def init_test_for_debug(test_module_name: str) -> Path:
    # Init logging
    logging.basicConfig(
            format="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s", 
            datefmt="%H:%M:%S", level=logging.INFO)

    # Prepare tmpdir
    tmp_basedir = Path(tempfile.gettempdir()) / test_module_name
    tmpdir = create_tempdir(parent_dir=tmp_basedir, base_dirname='debugrun')
    
    """
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    """

    return tmpdir
