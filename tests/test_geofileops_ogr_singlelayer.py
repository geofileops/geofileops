
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile
from geofileops.util import geofileops_ogr
from tests import test_helper

def test_select_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    basetest_select(input_path, output_path)

def test_select_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    basetest_select(input_path, output_path)
    
def basetest_select(
        input_path: Path, 
        output_path: Path):

    layerinfo_orig = geofile.getlayerinfo(input_path)
    sql_stmt = f'SELECT {layerinfo_orig.geometrycolumn}, oidn, uidn FROM "parcels"'
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

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_select_various_options_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    basetest_select_various_options(input_path, output_path)

def test_select_various_options_gpkg(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    basetest_select_various_options(input_path, output_path)
    
def basetest_select_various_options(
        input_path: Path, 
        output_path: Path):

    ### Check if columns parameter works (case insensitive) ###
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    layerinfo_orig = geofile.getlayerinfo(input_path)
    sql_stmt = f'''SELECT {layerinfo_orig.geometrycolumn}
                         {{columns_to_select_str}} 
                     FROM "parcels" '''
    geofileops_ogr.select(
            input_path=input_path,
            output_path=output_path,
            columns=columns,
            sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert 'OIDN' in layerinfo_select.columns
    assert 'UIDN' in layerinfo_select.columns
    assert len(layerinfo_select.columns) == len(columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Check if ... parameter works ###
    # TODO: increase test coverage of other options...

def test_isvalid_gpkg(tmpdir):
    # Buffer to test dir
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try both with and without gdal_bin set
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_bin')
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_default')

def test_isvalid_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_bin', ok_expected=True)
    basetest_isvalid(input_path, output_path, gdal_installation='gdal_default', ok_expected=True)
    
def basetest_isvalid(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str,
        ok_expected: bool = None):
    
    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        if ok_expected is None:
            ok_expected = test_helper.is_gdal_ok('isvalid', gdal_installation)
        try:
            assert geofileops_ogr.isvalid(input_path=input_path, output_path=output_path, nb_parallel=2) == True
            test_ok = True
        except:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    '''
    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert (len(layerinfo_orig.columns)+3) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    print(output_gdf)
    assert output_gdf['geom'][0] is None
    assert output_gdf['isvalid'][0] == 1
    assert output_gdf['isvalidreason'][0] == 'Valid Geometry'
    '''

def test_convexhull_gpkg(tmpdir):
    # Execute to test dir
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_bin')
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_default')

def test_convexhull_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = test_helper.get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    # Try both with and without gdal_bin set
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_bin')
    basetest_convexhull(input_path, output_path, gdal_installation='gdal_default')

def basetest_convexhull(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):
    
    # Do operation  
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
            geofileops_ogr.convexhull(input_path=input_path, output_path=output_path)
            test_ok = True
        except:
            test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.getlayerinfo(input_path)
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_buffer_gpkg(tmpdir):
    # Buffer to test dir
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try both with and without gdal_bin set
    basetest_buffer(input_path, output_path, gdal_installation='gdal_default')
    basetest_buffer(input_path, output_path, gdal_installation='gdal_bin')
        
def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    # Try both with and without gdal_bin set
    basetest_buffer(input_path, output_path, gdal_installation='gdal_default')
    basetest_buffer(input_path, output_path, gdal_installation='gdal_bin')
    
def basetest_buffer(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):
    
    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('buffer', gdal_installation)
        try:
            geofileops_ogr.buffer(input_path=input_path, output_path=output_path, distance=1)
            test_ok = True
        except:
            test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    layerinfo_orig = geofile.getlayerinfo(input_path)
    layerinfo_select = geofile.getlayerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None
    geofile.remove(output_path)

def test_makevalid_gpkg(tmpdir):
    # makevalid to test dir
    input_path = test_helper.get_testdata_dir() / 'invalid_geometries.gpkg'
    output_path = Path(tmpdir) / f"{input_path.stem}_valid.gpkg"
    
    # Try both with and without gdal_bin set
    basetest_makevalid(input_path, output_path, gdal_installation='gdal_bin')
    basetest_makevalid(input_path, output_path, gdal_installation='gdal_default')
           
def basetest_makevalid(
        input_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('makevalid', gdal_installation)
        try: 
            geofileops_ogr.makevalid(input_path=input_path, output_path=output_path, nb_parallel=2)
            test_ok = True
        except:
            test_ok = False
        assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

        # If it is expected not to be OK, don't do other checks
        if ok_expected is False:
            return

        # Now check if the output file is correctly created
        assert output_path.exists() == True
        layerinfo_orig = geofile.getlayerinfo(input_path)
        layerinfo_output = geofile.getlayerinfo(output_path)
        assert layerinfo_orig.featurecount == layerinfo_output.featurecount
        assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

        output_gdf = geofile.read_file(output_path)
        assert output_gdf['geometry'][0] is not None

        # Make sure the input file was not valid
        output_isvalid_path = output_path.parent / f"{output_path.stem}_isvalid{output_path.suffix}"
        isvalid = geofileops_ogr.isvalid(input_path=input_path, output_path=output_isvalid_path)
        assert isvalid is False, "Input file should contain invalid features"

        # Check if the result file is valid
        output_new_isvalid_path = output_path.parent / f"{output_path.stem}_new_isvalid{output_path.suffix}"
        isvalid = geofileops_ogr.isvalid(input_path=output_path, output_path=output_new_isvalid_path)
        assert isvalid == True, "Output file shouldn't contain invalid features"

def test_simplify_gpkg(tmpdir):
    # Simplify to test dir
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / input_path.name

    # Try both with and without gdal_bin set
    basetest_simplify(input_path, output_path, gdal_installation='gdal_default')
    basetest_simplify(input_path, output_path, gdal_installation='gdal_bin')
        
def test_simplify_shp(tmpdir):
    # Simplify to test dir
    input_path = test_helper.get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    # Try both with and without gdal_bin set
    basetest_simplify(input_path, output_path, gdal_installation='gdal_default')
    basetest_simplify(input_path, output_path, gdal_installation='gdal_bin')
    
def basetest_simplify(
        input_path: Path, 
        output_basepath: Path,
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
            geofileops_ogr.simplify(
                    input_path=input_path, output_path=output_path,
                    tolerance=5)
            test_ok = True
        except:
            test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.getlayerinfo(input_path)
    layerinfo_output = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

if __name__ == '__main__':
    import tempfile
    import shutil
    tmpdir = Path(tempfile.gettempdir()) / 'test_geofileops_ogr_ogr'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    
    # Single layer operations
    #test_buffer_gpkg(tmpdir)
    #test_makevalid_gpkg(tmpdir)
    #test_erase_shp(tmpdir)
    test_isvalid_shp(tmpdir)
    #test_isvalid_gpkg(tmpdir)
    #test_convexhull_shp(tmpdir)
    #test_convexhull_gpkg(tmpdir)
    #test_select_geos_version(tmpdir)
