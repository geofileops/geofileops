
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile
from geofileops.util import geofileops_ogr

class GdalBin():
    def __init__(self, set_gdal_bin: bool, gdal_bin_path: str = None):
        self.set_gdal_bin = set_gdal_bin
        if set_gdal_bin is True:
            if gdal_bin_path is None:
                self.gdal_bin_path = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
            else:
                self.gdal_bin_path = gdal_bin_path

    def __enter__(self):
        if self.set_gdal_bin is True:
            import os
            os.environ['GDAL_BIN'] = self.gdal_bin_path

    def __exit__(self, type, value, traceback):
        #Exception handling here
        import os
        if os.environ['GDAL_BIN'] is not None:
            del os.environ['GDAL_BIN']

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def is_gdal_default_ok() -> bool:
    return False

def is_gdal_bin_ok() -> bool:
    return True

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

def test_isvalid_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try with and without gdal_bin
    basetest_isvalid(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_isvalid(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def test_isvalid_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Try with and without gdal_bin
    basetest_isvalid(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_isvalid(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
    
def basetest_isvalid(
        input_path: Path, 
        output_path: Path, 
        set_gdal_bin: bool, 
        ok_expected: bool):
    
    # Do operation
    try:
        with GdalBin(set_gdal_bin):
            assert geofileops_ogr.isvalid(input_path=input_path, output_path=output_path) == True
        test_ok = True
    except:
        test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

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
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Without gdal_bin set, this fails at the moment
    basetest_convexhull(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_convexhull(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def test_convexhull_shp(tmpdir):
    # Select some data from src to tmp file
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    # Without gdal_bin set, this fails at the moment
    basetest_convexhull(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_convexhull(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def basetest_convexhull(
        input_path: Path, 
        output_path: Path, 
        set_gdal_bin: bool, 
        ok_expected: bool):
    
    # Do operation  
    try:
        with GdalBin(set_gdal_bin):
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
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Without gdal_bin set, this fails at the moment
    basetest_buffer(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_buffer(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
        
def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    # Without gdal_bin set, this fails at the moment
    basetest_buffer(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_buffer(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def basetest_buffer(
        input_path: Path, 
        output_path: Path, 
        set_gdal_bin: bool, 
        ok_expected: bool):
    
    # Do operation
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.buffer(input_path=input_path, output_path=output_path, distance=1)
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
    layerinfo_select = geofile.getlayerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None
    geofile.remove(output_path)

def test_makevalid_gpkg(tmpdir):
    # makevalid to test dir
    input_path = get_testdata_dir() / 'invalid_geometries.gpkg'
    output_path = Path(tmpdir) / input_path.name

    # Without gdal_bin set, this fails at the moment
    basetest_makevalid(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_makevalid(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
           
def basetest_makevalid(
        input_path: Path, 
        output_path: Path, 
        set_gdal_bin: bool, 
        ok_expected: bool):

    # Do operation
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.makevalid(input_path=input_path, output_path=output_path)
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
    with GdalBin(set_gdal_bin):
        isvalid = geofileops_ogr.isvalid(input_path=input_path, output_path=output_isvalid_path)
    assert isvalid is False, "Input file should contain invalid features"

    # Check if the result file is valid
    output_new_isvalid_path = output_path.parent / f"{output_path.stem}_new_isvalid{output_path.suffix}"
    with GdalBin(set_gdal_bin):
        isvalid = geofileops_ogr.isvalid(input_path=output_path, output_path=output_new_isvalid_path)
    assert isvalid == True, "Output file shouldn't contain invalid features"

def test_simplify_gpkg(tmpdir):
    # Simplify to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / input_path.name

    # Without gdal_bin set, this fails with libspatialite 4.3
    basetest_simplify(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_simplify(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
        
def test_simplify_shp(tmpdir):
    # Simplify to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'

    # Without gdal_bin set, this fails with libspatialite 4.3
    basetest_simplify(input_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_simplify(input_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
    
def basetest_simplify(
        input_path: Path, 
        output_path: Path,
        set_gdal_bin: bool, 
        ok_expected: bool):

    # Do operation
    try:
        with GdalBin(set_gdal_bin):
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

def test_erase_gpkg(tmpdir):
    # Erase to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    erase_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Without gdal_bin set, this fails with libspatialite 4.3
    basetest_erase(input_path, erase_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_erase(input_path, erase_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def test_erase_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    erase_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Without gdal_bin set, this fails with libspatialite 4.3
    basetest_erase(input_path, erase_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_erase(input_path, erase_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def basetest_erase(
        input_path: Path,
        erase_path: Path, 
        output_path: Path, 
        set_gdal_bin: bool, 
        ok_expected: bool):

    # Do operation
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.erase(
                    input_path=input_path, erase_path=erase_path,
                    output_path=output_path)
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

    # Without gdal_bin set, this fails with libspatialite 4.3
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=False, ok_expected=is_gdal_default_ok())
        
def test_export_by_location_shp(tmpdir):
    # Export to test dir
    input_to_select_from_path = get_testdata_dir() / 'parcels.shp'
    input_to_compare_with_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Without gdal_bin set, this fails at the moment
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def basetest_export_by_location(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path, 
        set_gdal_bin: bool, 
        ok_expected: bool):

    # Do operation
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.export_by_location(
                    input_to_select_from_path=input_to_select_from_path,
                    input_to_compare_with_path=input_to_compare_with_path,
                    output_path=output_path)
        test_ok = True
    except:
        test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.getlayerinfo(input_to_select_from_path)
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
    
    # Without gdal_bin set, this fails at the moment
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=False, ok_expected=is_gdal_default_ok())
    
def test_export_by_distance_shp(tmpdir):
    # Export to test dir
    input_to_select_from_path = get_testdata_dir() / 'parcels.shp'
    input_to_compare_with_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Without gdal_bin set, this fails at the moment
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            set_gdal_bin=False, ok_expected=is_gdal_default_ok())

def basetest_export_by_distance(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path,
        set_gdal_bin: bool, 
        ok_expected: bool):

    # Do operation
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.export_by_distance(
                    input_to_select_from_path=input_to_select_from_path,
                    input_to_compare_with_path=input_to_compare_with_path,
                    max_distance=10,
                    output_path=output_path)
        test_ok = True
    except:
        test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.getlayerinfo(input_to_select_from_path)
    layerinfo_select = geofile.getlayerinfo(input_to_select_from_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_intersect_gpkg(tmpdir):
    # Export to test dir
    input1_path = get_testdata_dir() / 'parcels.gpkg'
    input2_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_intersect_zones.gpkg'

    # Without gdal_bin set, this fails at the moment
    basetest_intersect(input1_path, input2_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
    basetest_intersect(input1_path, input2_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    
def test_intersect_shp(tmpdir):
    # Export to test dir
    input1_path = get_testdata_dir() / 'parcels.shp'
    input2_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_intersect_zones.gpkg'
    
    # Without gdal_bin set, this fails at the moment
    basetest_intersect(input1_path, input2_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
    basetest_intersect(input1_path, input2_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    
def basetest_intersect(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path, 
        set_gdal_bin: bool, 
        ok_expected: bool):

    # Do operation
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.intersect(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    verbose=True)
        test_ok = True
    except:
        test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.getlayerinfo(input1_path)
    layerinfo_input2 = geofile.getlayerinfo(input2_path)
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_select.featurecount == 28
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_select.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_join_by_location_gpkg(tmpdir):
    # Export to test dir
    input1_path = get_testdata_dir() / 'parcels.gpkg'
    input2_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Without gdal_bin set, this fails at the moment
    basetest_join_by_location(input1_path, input2_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
    basetest_join_by_location(input1_path, input2_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    
def test_join_by_location_shp(tmpdir):
    # Export to test dir
    input1_path = get_testdata_dir() / 'parcels.shp'
    input2_path = get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Without gdal_bin set, this fails at the moment
    basetest_join_by_location(input1_path, input2_path, output_path, set_gdal_bin=False, ok_expected=is_gdal_default_ok())
    basetest_join_by_location(input1_path, input2_path, output_path, set_gdal_bin=True, ok_expected=is_gdal_bin_ok())
    
def basetest_join_by_location(
        input1_path: Path, input2_path: Path, output_path: Path, 
        set_gdal_bin: bool, ok_expected: bool):
        
    ### Test 1: inner join, intersect
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.join_by_location(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    force=True)
        test_ok = True
    except:
        test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.getlayerinfo(input1_path)
    layerinfo_input2 = geofile.getlayerinfo(input2_path)
    layerinfo_result = geofile.getlayerinfo(output_path)
    assert layerinfo_result.featurecount == 4
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 1) == len(layerinfo_result.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Test 2: left outer join, intersect
    try:
        with GdalBin(set_gdal_bin):
            geofileops_ogr.join_by_location(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    discard_nonmatching=False,
                    force=True)
        test_ok = True
    except:
        test_ok = False
    assert test_ok is ok_expected, "Without gdal_bin set to an osgeo installation, it is 'normal' this fails"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_result = geofile.getlayerinfo(output_path)
    assert layerinfo_result.featurecount == 24, f"Featurecount is {layerinfo_result.featurecount}, expected 48"
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 1) == len(layerinfo_result.columns)

    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

if __name__ == '__main__':
    import tempfile
    import shutil
    tmpdir = Path(tempfile.gettempdir()) / 'test_geofileops_ogr_ogr'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    
    # Single layer operations
    test_buffer_gpkg(tmpdir)
    #test_makevalid_gpkg(tmpdir)
    #test_erase_shp(tmpdir)
    #test_isvalid_shp(tmpdir)
    #test_isvalid_gpkg(tmpdir)
    #test_convexhull_shp(tmpdir)
    #test_convexhull_gpkg(tmpdir)
    #test_select_geos_version(tmpdir)

    # Two layer operations
    #test_intersect_gpkg(tmpdir)
    #test_export_by_distance_shp(tmpdir)
    #test_join_by_location_gpkg(tmpdir)
    