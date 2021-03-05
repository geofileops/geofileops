# -*- coding: utf-8 -*-
"""
Tests for operations using ogr on two layers.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.geofile import GeometryType
from geofileops.util import geofileops_ogr
from geofileops.util import ogr_util
from tests import test_helper

def test_erase_gpkg(tmpdir):
    # Erase from polygon layer, with and without gdal_bin set
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    erase_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_erase_zones.gpkg'
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_bin')
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_default')

    # Erase from point layer, with and without gdal_bin set
    input_path = test_helper.get_testdata_dir() / 'points.gpkg'
    erase_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'points_erase_zones.gpkg'
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOINT,
            gdal_installation='gdal_default')
    basetest_erase(input_path, erase_path, output_path,
            expected_output_geometrytype=GeometryType.MULTIPOINT,
            gdal_installation='gdal_bin')

    # Erase from line layer, with and without gdal_bin set
    input_path = test_helper.get_testdata_dir() / 'linestrings_rows_of_trees.gpkg'
    erase_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'rows_of_trees_erase_zones.gpkg'
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTILINESTRING,
            gdal_installation='gdal_default')
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTILINESTRING,
            gdal_installation='gdal_bin')

def test_erase_shp(tmpdir):
    # Prepare input and output paths
    input_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    erase_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_erase_zones.shp'

    # Try both with and without gdal_bin set
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_bin')
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON,
            gdal_installation='gdal_default')

def basetest_erase(
        input_path: Path,
        erase_path: Path, 
        output_basepath: Path,
        expected_output_geometrytype: GeometryType, 
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.erase(
                    input_path=input_path, erase_path=erase_path,
                    output_path=output_path)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Checks depending on geometry type
    assert layerinfo_output.geometrytype == expected_output_geometrytype
    if expected_output_geometrytype == GeometryType.MULTIPOLYGON:
        assert layerinfo_output.featurecount == 37
    elif expected_output_geometrytype == GeometryType.MULTIPOINT:
        assert layerinfo_output.featurecount == 47
    elif expected_output_geometrytype == GeometryType.MULTILINESTRING:
        assert layerinfo_output.featurecount == 12
    else:
        raise Exception(f"Unsupported expected_output_geometrytype: {expected_output_geometrytype}")

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_export_by_location_gpkg(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_loc_zones.gpkg'

    # Try both with and without gdal_bin set
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_bin')
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_default')
        
def test_export_by_location_shp(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_loc_zones.shp'

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
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.export_by_location(
                    input_to_select_from_path=input_to_select_from_path,
                    input_to_compare_with_path=input_to_compare_with_path,
                    output_path=output_path)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_to_select_from_path)
    layerinfo_output = geofile.get_layerinfo(input_to_select_from_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_export_by_distance_gpkg(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_distance_zones.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_bin')
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path, 
            gdal_installation='gdal_default')
    
def test_export_by_distance_shp(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    input_to_compare_with_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_distance_zones.shp'

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
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.export_by_distance(
                    input_to_select_from_path=input_to_select_from_path,
                    input_to_compare_with_path=input_to_compare_with_path,
                    max_distance=10,
                    output_path=output_path)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_to_select_from_path)
    layerinfo_output = geofile.get_layerinfo(input_to_select_from_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_intersect_gpkg(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_intersect_zones.gpkg'

    # Try both with and without gdal_bin set
    basetest_intersect(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_intersect(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def test_intersect_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
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
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.intersect(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    verbose=True)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 28
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)

    # Check geometry type
    if output_path.suffix.lower() == '.shp':
        # For shapefiles the type stays POLYGON anyway 
        assert layerinfo_output.geometrytype == GeometryType.POLYGON 
    elif output_path.suffix.lower() == '.gpkg':
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_join_by_location_gpkg(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_join_zones.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_join_by_location(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_join_by_location(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def test_join_by_location_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_join_zones.gpkg'
    
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
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.join_by_location(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    discard_nonmatching=True,
                    force=True)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 28
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 1) == len(layerinfo_output.columns)

    # Check geometry type
    if output_path.suffix.lower() == '.shp':
        # For shapefiles the type stays POLYGON anyway 
        assert layerinfo_output.geometrytype == GeometryType.POLYGON 
    elif output_path.suffix.lower() == '.gpkg':
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Test 2: left outer join, intersect
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}_test2_{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.join_by_location(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    discard_nonmatching=False,
                    force=True)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 48
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 1) == len(layerinfo_output.columns)

    # Check geometry type
    if output_path.suffix.lower() == '.shp':
        # For shapefiles the type stays POLYGON anyway 
        assert layerinfo_output.geometrytype == GeometryType.POLYGON 
    elif output_path.suffix.lower() == '.gpkg':
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_split_gpkg(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_split_zones.gpkg'

    # Try both with and without gdal_bin set
    basetest_split(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_split(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def test_split_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_split_zones.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_split(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_split(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def basetest_split(
        input1_path: Path, 
        input2_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.split(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    verbose=True)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 63
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)

    # Check geometry type
    if output_path.suffix.lower() == '.shp':
        # For shapefiles the type stays POLYGON anyway 
        assert layerinfo_output.geometrytype == GeometryType.POLYGON 
    elif output_path.suffix.lower() == '.gpkg':
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_union_gpkg(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.gpkg'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_union_zones.gpkg'

    # Try both with and without gdal_bin set
    basetest_union(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_union(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def test_union_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.get_testdata_dir() / 'polygons_parcels.shp'
    input2_path = test_helper.get_testdata_dir() / 'polygons_zones.gpkg'
    output_path = Path(tmpdir) / 'parcels_union_zones.gpkg'
    
    # Try both with and without gdal_bin set
    basetest_union(input1_path, input2_path, output_path, gdal_installation='gdal_default')
    basetest_union(input1_path, input2_path, output_path, gdal_installation='gdal_bin')
    
def basetest_union(
        input1_path: Path, 
        input2_path: Path, 
        output_basepath: Path, 
        gdal_installation: str):

    # Do operation
    output_path = output_basepath.parent / f"{output_basepath.stem}_{gdal_installation}{output_basepath.suffix}"
    with test_helper.GdalBin(gdal_installation):
        ok_expected = test_helper.is_gdal_ok('twolayer', gdal_installation)
        try:
            geofileops_ogr.union(
                    input1_path=input1_path,
                    input2_path=input2_path,
                    output_path=output_path,
                    verbose=True)
            test_ok = True
        except ogr_util.SQLNotSupportedException:
            test_ok = False
    assert test_ok is ok_expected, f"Error: for {gdal_installation}, test_ok: {test_ok}, expected: {ok_expected}"

    # If it is expected not to be OK, don't do other checks
    if ok_expected is False:
        return

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 67
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)

    # Check geometry type
    if output_path.suffix.lower() == '.shp':
        # For shapefiles the type stays POLYGON anyway 
        assert layerinfo_output.geometrytype == GeometryType.POLYGON 
    elif output_path.suffix.lower() == '.gpkg':
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Two layer operations
    #test_erase_gpkg(tmpdir)
    #test_intersect_gpkg(tmpdir)
    #test_export_by_distance_shp(tmpdir)
    #test_join_by_location_gpkg(tmpdir)
    test_split_gpkg(tmpdir)
    #test_union_gpkg(tmpdir)
    