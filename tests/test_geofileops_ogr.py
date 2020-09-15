
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile
from geofileops.util import geofileops_ogr

class GdalBin():
    def __init__(self, gdal_bin_path: str = 'DEFAULT'):
        if gdal_bin_path is not None:
            if gdal_bin_path.upper() == 'DEFAULT':
                self.gdal_bin_path = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
            else:
                self.gdal_bin_path = gdal_bin_path

    def __enter__(self):
        if self.gdal_bin_path is not None:
            import os
            os.environ['GDAL_BIN'] = self.gdal_bin_path

    def __exit__(self, type, value, traceback):
        #Exception handling here
        import os
        if os.environ['GDAL_BIN'] is not None:
            del os.environ['GDAL_BIN']

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

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

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

def test_check_valid_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_check_valid(input_path, output_path)

    # Without gdal_bin set, this fails at the moment
    try:
        basetest_check_valid(input_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
        
def test_check_valid_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_check_valid(input_path, output_path)

    # Without gdal_bin set, this fails at the moment
    try:
        basetest_check_valid(input_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

def basetest_check_valid(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_ogr.check_valid(
            input_path=input_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert (len(layerinfo_orig.columns)+3) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    print(output_gdf)
    assert output_gdf['isvalid'][0] == 1
    assert output_gdf['isvalidreason'][0] == 'Valid Geometry'
    assert output_gdf['isvaliddetail'][0] is None
    
def test_convexhull_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_convexhull(input_path, output_path)

    # Without gdal_bin set, this fails at the moment
    try:
        basetest_convexhull(input_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    
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

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_buffer_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_buffer(input_path, output_path)

    # Without gdal_bin set, this fails at the moment
    try:
        basetest_buffer(input_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
        
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

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_simplify_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_simplify(input_path, output_path)

    # Without gdal_bin set, this fails at the moment
    try:
        basetest_simplify(input_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
        
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

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_erase_gpkg(tmpdir):
    # Erase to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    erase_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_erase(input_path, erase_path, output_path)

def test_erase_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    erase_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_erase(input_path, erase_path, output_path)

def basetest_erase(input_path, erase_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_ogr.erase(
            input_path=input_path,
            erase_path=erase_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_export_by_location_gpkg(tmpdir):
    # Export to test dir
    input_to_select_from_path = get_testdata_dir() / 'parcels.gpkg'
    input_to_compare_with_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_export_by_location(input_to_select_from_path, input_to_compare_with_path, output_path)

    # Without gdal_bin set, this fails at the moment
    try:
        basetest_export_by_location(input_to_select_from_path, input_to_compare_with_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

def test_export_by_location_shp(tmpdir):
    # Export to test dir
    input_to_select_from_path = get_testdata_dir() / 'parcels.shp'
    input_to_compare_with_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_export_by_location(input_to_select_from_path, input_to_compare_with_path, output_path)

    # Without gdal_bin set, this fails at the moment
    try:
        basetest_export_by_location(input_to_select_from_path, input_to_compare_with_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

def basetest_export_by_location(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path):
    layerinfo_orig = geofile.getlayerinfo(input_to_select_from_path)
    geofileops_ogr.export_by_location(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input_to_select_from_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_export_by_distance_gpkg(tmpdir):
        # Export to test dir
    input_to_select_from_path = get_testdata_dir() / 'parcels.gpkg'
    input_to_compare_with_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_export_by_location(input_to_select_from_path, input_to_compare_with_path, output_path)
    
    # Without gdal_bin set, this fails at the moment
    try:
        basetest_export_by_location(input_to_select_from_path, input_to_compare_with_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    
def test_export_by_distance_shp(tmpdir):
    # Export to test dir
    input_to_select_from_path = get_testdata_dir() / 'parcels.shp'
    input_to_compare_with_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_export_by_distance(input_to_select_from_path, input_to_compare_with_path, output_path)
    
    # Without gdal_bin set, this fails at the moment
    try:
        basetest_export_by_distance(input_to_select_from_path, input_to_compare_with_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    
def basetest_export_by_distance(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path):
    layerinfo_orig = geofile.getlayerinfo(input_to_select_from_path)
    geofileops_ogr.export_by_distance(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            max_distance=10,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input_to_select_from_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_intersect_gpkg(tmpdir):
    # Export to test dir
    input1_path = get_testdata_dir() / 'parcels.gpkg'
    input2_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_intersect(input1_path, input2_path, output_path)
    
    # Without gdal_bin set, this fails at the moment
    try:
        basetest_intersect(input1_path, input2_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    
def test_intersect_shp(tmpdir):
    # Export to test dir
    input1_path = get_testdata_dir() / 'parcels.shp'
    input2_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    with GdalBin():
        basetest_intersect(input1_path, input2_path, output_path)
    
    # Without gdal_bin set, this fails at the moment
    try:
        basetest_intersect(input1_path, input2_path, output_path)
        assert True is False, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    except:
        assert True is True, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"
    
def basetest_intersect(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):
        
    layerinfo_orig = geofile.getlayerinfo(input1_path)
    geofileops_ogr.intersect(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input1_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

if __name__ == '__main__':
    import tempfile
    import shutil
    tmpdir = Path(tempfile.gettempdir()) / 'test_geofileops_ogr_ogr'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    #test_buffer_gpkg(tmpdir)
    #test_erase_shp(tmpdir)
    test_intersect_gpkg(tmpdir)
    #test_export_by_distance_shp(tmpdir)
    #test_check_valid_shp(tmpdir)
    #test_convexhull_shp(tmpdir)
    #test_select_geos_version(tmpdir)