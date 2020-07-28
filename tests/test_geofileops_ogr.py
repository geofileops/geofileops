
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile
from geofileops.util import geofileops_ogr

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_select_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    layerinfo_orig = geofile.getlayerinfo(input_path)
    sql_stmt = f'select {layerinfo_orig.geometrycolumn}, oidn, uidn from "parcels"'
    geofileops_ogr.select(
            input_path=input_path,
            output_path=output_path,
            sql_stmt=sql_stmt,
            sql_dialect='SQLITE')

    # Now check if the tmp file is correctly created
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert 'OIDN' in layerinfo_select.columns
    assert 'UIDN' in layerinfo_select.columns
    assert len(layerinfo_select.columns) == 2

def test_select_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    layerinfo_orig = geofile.getlayerinfo(input_path)
    sql_stmt = f'select {layerinfo_orig.geometrycolumn}, oidn, uidn from "parcels"'
    geofileops_ogr.select(
            input_path=input_path,
            output_path=output_path,
            sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert 'OIDN' in layerinfo_select.columns
    assert 'UIDN' in layerinfo_select.columns
    assert len(layerinfo_select.columns) == 2

def test_select_geos_verion(tmpdir):
    # Get some version info from spatialite...
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    sql_stmt = f'select spatialite_version(), geos_version(), lwgeom_version() from "parcels" limit 1'
    geofileops_ogr.select(
            input_path=input_path,
            output_path=output_path,
            sql_stmt=sql_stmt)
    result_gdf = geofile.read_file(output_path)

    assert 'GDAL_BIN' not in os.environ    
    assert result_gdf['spatialite_version()'][0] == '4.3.0-RC1'
    assert result_gdf['geos_version()'][0] >= '3.8.1'
    assert result_gdf['lwgeom_version()'][0] is not None

def test_convexhull_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_convexhull(input_path, output_path)

def test_convexhull_gpkg_osgeo4w(tmpdir):
    import os
    os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"

    try:
        # Buffer to test dir
        input_path = get_testdata_dir() / 'parcels.gpkg'
        output_path = Path(tmpdir) / 'parcels.gpkg'
        basetest_convexhull(input_path, output_path)
    finally:
        del os.environ['GDAL_BIN']

def test_convexhull_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_convexhull(input_path, output_path)

def basetest_convexhull(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_ogr.convexhull(
            input_path=input_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

def test_buffer_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_buffer(input_path, output_path)

def test_buffer_gpkg_osgeo4w(tmpdir):
    import os
    os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"

    try:
        # Buffer to test dir
        input_path = get_testdata_dir() / 'parcels.gpkg'
        output_path = Path(tmpdir) / 'parcels.gpkg'
        basetest_buffer(input_path, output_path)
    finally:
        del os.environ['GDAL_BIN']

def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_buffer(input_path, output_path)

def basetest_buffer(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_ogr.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=1)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

def test_simplify_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_simplify(input_path, output_path)

def test_simplify_gpkg_osgeo4w(tmpdir):
    import os
    os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"

    try:
        # Buffer to test dir
        input_path = get_testdata_dir() / 'parcels.gpkg'
        output_path = Path(tmpdir) / 'parcels.gpkg'
        basetest_simplify(input_path, output_path)
    finally:
        del os.environ['GDAL_BIN']

def test_simplify_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_simplify(input_path, output_path)

def basetest_simplify(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_ogr.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=5)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

if __name__ == '__main__':
    import tempfile
    import shutil
    tmpdir = Path(tempfile.gettempdir()) / 'test_geofileops_ogr_ogr'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    test_buffer_gpkg(tmpdir)
