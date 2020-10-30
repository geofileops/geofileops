
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile
from geofileops.util import geofileops_ogr
from tests import test_helper

def test_erase_gpkg(tmpdir):
    # Prepare input and output paths
    input_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    erase_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try both with and without gdal_bin set
    basetest_erase(input_path, erase_path, output_path, gdal_installation='gdal_bin')
    basetest_erase(input_path, erase_path, output_path, gdal_installation='gdal_default')

def test_erase_shp(tmpdir):
    # Prepare input and output paths
    input_path = test_helper.get_testdata_dir() / 'parcels.shp'
    erase_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try both with and without gdal_bin set
    basetest_erase(input_path, erase_path, output_path, gdal_installation='gdal_bin')
    basetest_erase(input_path, erase_path, output_path, gdal_installation='gdal_default')

def basetest_erase(
        input_path: Path,
        erase_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
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
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try both with and without gdal_bin set
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_bin')
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_default')
        
def test_export_by_location_shp(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'parcels.shp'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try both with and without gdal_bin set
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_bin')
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_default')

def basetest_export_by_location(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
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
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_bin')
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_default')
    
def test_export_by_distance_shp(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'parcels.shp'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'

    # Try both with and without gdal_bin set
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_bin')
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_default')

def basetest_export_by_distance(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_basepath: Path,
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
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
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    input2_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_intersect_zones.gpkg'

    # Try both with and without gdal_bin set
    basetest_intersect(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_intersect(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def test_intersect_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'parcels.shp'
    input2_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_intersect_zones.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_intersect(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_intersect(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def basetest_intersect(
        input1_path: Path, 
        input2_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
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
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'parcels.gpkg'
    input2_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_join_by_location(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_join_by_location(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def test_join_by_location_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'parcels.shp'
    input2_path = test_helper.get_testdata_dir() / 'zones.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_join_by_location(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_join_by_location(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def basetest_join_by_location(
        input1_path: Path, input2_path: Path,
        output_basepath: Path, 
        gdal_installation: str):
        
    ### Test 1: inner join, intersect
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}_test1_{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
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
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}_test2_{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('', gdal_installation)
        try:
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

    # Two layer operations
    test_erase_gpkg(tmpdir)
    #test_intersect_gpkg(tmpdir)
    #test_export_by_distance_shp(tmpdir)
    #test_join_by_location_gpkg(tmpdir)
    