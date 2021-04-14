# -*- coding: utf-8 -*-
"""
Helper functions for all tests.
"""

from geofileops.util.general_util import MissingRuntimeDependencyError
import logging
import os
from pathlib import Path
import sys
import tempfile

import shapely.geometry as sh_geom

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import ogr_util
from geofileops.util import sqlite_util

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
            os.environ['GDAL_BIN'] = self.gdal_bin_path
            curr_script_dir = Path(__file__).resolve().parent
            mod_spatialite_dir = None
            if os.name == 'nt':
                mod_spatialite_dir = curr_script_dir / 'mod_spatialite' / 'mod_spatialite-5.0.1-win-amd64' 
            else: 
                raise Exception(f"os.name not supported: {os.name}")
            if mod_spatialite_dir is not None:
                os.environ['MOD_SPATIALITE_DIR'] = str(mod_spatialite_dir)
        else:
            if os.environ.get('MOD_SPATIALITE_DIR') is not None:
                del os.environ['MOD_SPATIALITE_DIR']

    def __exit__(self, type, value, traceback):
        #Exception handling here
        if os.environ.get('GDAL_BIN') is not None:
            del os.environ['GDAL_BIN']
        if os.environ.get('MOD_SPATIALITE_DIR') is not None:
            del os.environ['MOD_SPATIALITE_DIR']

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

def check_runtime_dependencies_ok(operation: str, gdal_installation: str) -> bool:
    # Operations on two layers use sqlite directly to run sql -> check sqlite
    if operation in ['twolayer']:
        try:
            sqlite_util.check_runtimedependencies()
            return True
        except MissingRuntimeDependencyError:
            return False
    else:
        # Operations that use gdal to run sql -> check gdal
        # Check if there are unsupported functions
        install_info = ogr_util.get_gdal_install_info(gdal_installation)
        if install_info['spatialite_version()'] >= '5.0.0':
            return True
        elif install_info['spatialite_version()'] >= '4.3.0':
            if install_info['lwgeom_version()'] is None:
                if operation in ['twolayer']:
                    return False
                else:
                    return True
            else:
                return True
        return False
        
