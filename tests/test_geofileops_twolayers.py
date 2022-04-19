# -*- coding: utf-8 -*-
"""
Tests for operations that are executed using a sql statement on two layers.
"""

from pathlib import Path
import sys

import pytest

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geofileops as gfo
from geofileops import GeometryType, PrimitiveType
from geofileops.util import _io_util
from geofileops.util import _geoops_sql
from tests import test_helper

def test_clip(tmpdir):
    # Init
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    test_inputs = []
    test_inputs.append({
            "input_path": test_helper.TestFiles.polygons_parcels_gpkg,
            "geometrytype": GeometryType.MULTIPOLYGON})
    test_inputs.append({
            "input_path": test_helper.TestFiles.points_gpkg,
            "geometrytype": GeometryType.MULTIPOINT})
    test_inputs.append({
            "input_path": test_helper.TestFiles.linestrings_rows_of_trees_gpkg,
            "geometrytype": GeometryType.MULTILINESTRING})

    # Prepare test data + run tests
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            for test_input in test_inputs: 
                # If test input file is in wrong format, convert it
                input_path = test_helper.prepare_test_file(
                        input_path=test_input['input_path'],
                        output_dir=tmp_dir,
                        suffix=suffix,
                        crs_epsg=crs_epsg)

                # If test input file is in wrong format, convert it
                clip_path = test_helper.prepare_test_file(
                        input_path=test_helper.TestFiles.polygons_zones_gpkg,
                        output_dir=tmp_dir,
                        suffix=suffix,
                        crs_epsg=crs_epsg)
            
                # Now run test
                output_path = tmp_dir / f"{input_path.stem}-output{suffix}"
                print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}, geometrytype {test_input['geometrytype']}")
                basetest_clip(input_path, clip_path, output_path, test_input['geometrytype'])

def basetest_clip(
        input_path: Path,
        clip_path: Path, 
        output_path: Path,
        expected_output_geometrytype: GeometryType):

    ### Do standard operation ###
    gfo.clip(
            input_path=input_path, 
            clip_path=clip_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = gfo.get_layerinfo(input_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Checks depending on geometry type
    assert layerinfo_output.geometrytype == expected_output_geometrytype
    if expected_output_geometrytype == GeometryType.MULTIPOLYGON:
        assert layerinfo_output.featurecount == 26
    elif expected_output_geometrytype == GeometryType.MULTIPOINT:
        assert layerinfo_output.featurecount == 3
    elif expected_output_geometrytype == GeometryType.MULTILINESTRING:
        assert layerinfo_output.featurecount == 15
    else:
        raise Exception(f"Unsupported expected_output_geometrytype: {expected_output_geometrytype}")

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_erase(tmpdir):
    # Init
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    test_inputs = []
    test_inputs.append({
            "input_path": test_helper.TestFiles.polygons_parcels_gpkg,
            "geometrytype": GeometryType.MULTIPOLYGON})
    test_inputs.append({
            "input_path": test_helper.TestFiles.points_gpkg,
            "geometrytype": GeometryType.MULTIPOINT})
    test_inputs.append({
            "input_path": test_helper.TestFiles.linestrings_rows_of_trees_gpkg,
            "geometrytype": GeometryType.MULTILINESTRING})

    # Prepare test data + run tests
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            for test_input in test_inputs: 
                # If test input file is in wrong format, convert it
                input_path = test_helper.prepare_test_file(
                        input_path=test_input['input_path'],
                        output_dir=tmp_dir,
                        suffix=suffix,
                        crs_epsg=crs_epsg)

                # If test input file is in wrong format, convert it
                erase_path = test_helper.prepare_test_file(
                        input_path=test_helper.TestFiles.polygons_zones_gpkg,
                        output_dir=tmp_dir,
                        suffix=suffix,
                        crs_epsg=crs_epsg)
            
                # Now run test
                output_path = tmp_dir / f"{input_path.stem}-output{suffix}"
                print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}, geometrytype {test_input['geometrytype']}")
                basetest_erase(input_path, erase_path, output_path, test_input['geometrytype'])

def basetest_erase(
        input_path: Path,
        erase_path: Path, 
        output_path: Path,
        expected_output_geometrytype: GeometryType):

    ### Do standard operation ###
    gfo.erase(
            input_path=input_path, 
            erase_path=erase_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = gfo.get_layerinfo(input_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
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
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Do operation with explodecollections=True ###
    output_path = _io_util.with_stem(output_path, f"{output_path.stem}_exploded")
    gfo.erase(
            input_path=input_path, 
            erase_path=erase_path,
            output_path=output_path,
            explodecollections=True)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = gfo.get_layerinfo(input_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Checks depending on geometry type
    assert layerinfo_output.geometrytype == expected_output_geometrytype
    if expected_output_geometrytype == GeometryType.MULTIPOLYGON:
        assert layerinfo_output.featurecount == 40
    elif expected_output_geometrytype == GeometryType.MULTIPOINT:
        assert layerinfo_output.featurecount == 47
    elif expected_output_geometrytype == GeometryType.MULTILINESTRING:
        assert layerinfo_output.featurecount == 13
    else:
        raise Exception(f"Unsupported expected_output_geometrytype: {expected_output_geometrytype}")

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None
        
def test_export_by_location(tmpdir):
    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            # If test input file is in wrong format, convert it
            input_to_select_from_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_parcels_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)

            # If test input file is in wrong format, convert it
            input_to_compare_with_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_zones_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)
        
            # Now run test
            output_path = tmp_dir / f"{input_to_select_from_path.stem}-output{suffix}"
            print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
            basetest_export_by_location(input_to_select_from_path, input_to_compare_with_path, output_path)

def basetest_export_by_location(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path):

    gfo.export_by_location(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = gfo.get_layerinfo(input_to_select_from_path)
    layerinfo_output = gfo.get_layerinfo(input_to_select_from_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_export_by_distance(tmpdir):
    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            # If test input file is in wrong format, convert it
            input_to_select_from_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_parcels_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)

            # If test input file is in wrong format, convert it
            input_to_compare_with_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_zones_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)
        
            # Now run test
            output_path = tmp_dir / f"{input_to_select_from_path.stem}-output{suffix}"
            print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
            basetest_export_by_distance(
                    input_to_select_from_path, 
                    input_to_compare_with_path, 
                    output_path)

def basetest_export_by_distance(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path):

    gfo.export_by_distance(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            max_distance=10,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = gfo.get_layerinfo(input_to_select_from_path)
    layerinfo_output = gfo.get_layerinfo(input_to_select_from_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_intersection(tmpdir):
    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            # If test input file is in wrong format, convert it
            input1_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_parcels_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)

            # If test input file is in wrong format, convert it
            input2_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_zones_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)
        
            # Now run test
            output_path = tmp_dir / f"{input1_path.stem}-output{suffix}"
            print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
            basetest_intersection(input1_path, input2_path, output_path)
    
def basetest_intersection(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Do operation
    gfo.intersection(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            nb_parallel=2)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 29
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 
    
    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_prepare_spatial_relations_filter():
    # Test all existing named relations
    named_relations = ["equals", "touches", "within", "overlaps", "crosses", 
            "intersects", "contains", "covers", "coveredby"]
    for relation in named_relations:
        query = f"{relation} is True"
        filter = _geoops_sql._prepare_spatial_relations_filter(query)
        assert filter is not None and filter != ""
    
    # Test extra queries that should work
    ok_queries = [
            "intersects is False",
            "(intersects is False and within is True) and crosses is False"
            "(((T******** is False)))"]
    for query in ok_queries:
        filter = _geoops_sql._prepare_spatial_relations_filter(query)
        assert filter is not None and filter != ""

    # Test queries that should fail
    error_queries = [
            ("Intersects is False", "named relations should be in lowercase"),
            ("intersects Is False", "is should be in lowercase"),
            ("intersects is false", "false should be False"),
            ("intersects = false", "= should be is"),
            ("(intersects is False", "not all brackets are closed"),
            ("intersects is False)", "more closing brackets then opened ones"),
            ("T**T**T* is False", "predicate should be 9 characters, not 8"),
            ("T**T**T**T is False", "predicate should be 9 characters, not 10"),
            ("A**T**T** is False", "A is not a valid character in a predicate"),
            ("'T**T**T**' is False", "predicates should not be quoted"),
            ("[T**T**T** is False ]", "square brackets are not supported"),]
    for query, error_reason in error_queries:
        try:
            _ = _geoops_sql._prepare_spatial_relations_filter(query)
            error = False
        except:
            error = True
        assert error is True, error_reason

@pytest.mark.parametrize(
        "suffix, crs_epsg, spatial_relations_query, discard_nonmatching, min_area_intersect, expected_featurecount", 
        [   (".gpkg", 31370, "intersects is False", False, None, 46),
            (".gpkg", 31370, "intersects is False", True, None, 0),
            (".gpkg", 31370, "intersects is True", False, 1000, 48),
            (".gpkg", 31370, "intersects is True", False, None, 49),
            (".gpkg", 31370, "intersects is True", True, 1000, 25),
            (".gpkg", 31370, "intersects is True", True, None, 29), 
            (".gpkg", 31370, "T******** is True or *T******* is True", True, None, 29),
            (".gpkg", 4326, "intersects is True", False, None, 49),
            (".shp", 31370, "intersects is True", False, None, 49), ])
def test_join_by_location(
        tmpdir, 
        suffix: str,
        spatial_relations_query: str,
        crs_epsg: int,
        discard_nonmatching: bool,
        min_area_intersect: float,
        expected_featurecount: int):
    ### Prepare test data + run tests ###
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path=test_helper.TestFiles.polygons_parcels_gpkg
    input1_path = test_helper.prepare_test_file(path, tmp_dir, suffix, crs_epsg)
    path = test_helper.TestFiles.polygons_zones_gpkg
    input2_path = test_helper.prepare_test_file(path, tmp_dir, suffix, crs_epsg)

    ### Test join_by_location ###
    output_path = tmp_dir / f"{input1_path.stem}-output_{discard_nonmatching}_{min_area_intersect}{suffix}"
    gfo.join_by_location(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            spatial_relations_query=spatial_relations_query,
            discard_nonmatching=discard_nonmatching,
            min_area_intersect=min_area_intersect,
            force=True)

    # If no result expected, the output files shouldn't exist
    if expected_featurecount == 0:
        assert output_path.exists() is False
        return

    # Check if the output file is correctly created
    assert output_path.exists() is True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == expected_featurecount
    assert len(layerinfo_output.columns) == (
            len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 1)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

@pytest.mark.parametrize("suffix, crs_epsg", [(".gpkg", 31370), (".gpkg", 4326), (".shp", 31370)])
def test_join_nearest(tmpdir, suffix, crs_epsg):
    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # If test input file is in wrong format, convert it
    input1_path = test_helper.prepare_test_file(
            input_path=test_helper.TestFiles.polygons_parcels_gpkg,
            output_dir=tmp_dir,
            suffix=suffix,
            crs_epsg=crs_epsg)

    # If test input file is in wrong format, convert it
    input2_path = test_helper.prepare_test_file(
            input_path=test_helper.TestFiles.polygons_zones_gpkg,
            output_dir=tmp_dir,
            suffix=suffix,
            crs_epsg=crs_epsg)

    # Now run test
    output_path = tmp_dir / f"{input1_path.stem}-output{suffix}"
    print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
    nb_nearest = 2
    gfo.join_nearest(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            nb_nearest=nb_nearest,
            force=True)

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == nb_nearest * layerinfo_input1.featurecount
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 2) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

@pytest.mark.parametrize(
        "suffix, crs_epsg, overlay_operation, discard_nonmatching, expected_featurecount", 
        [   (".gpkg", 31370, "difference", False, 46),
            (".gpkg", 31370, "identity", True, 0),
            (".gpkg", 31370, "intersection", False, 48),
            (".gpkg", 31370, "symmetric_difference", False, 49),
            (".gpkg", 31370, "union", True, 25),
            (".gpkg", 4326, "intersection", False, 49),
            (".shp", 31370, "intersection", False, 49), ])
def test_overlay(
        tmpdir, 
        suffix: str,
        crs_epsg: int,
        overlay_operation: str,
        discard_nonmatching: bool,
        expected_featurecount: int):
    ### Prepare test data + run tests ###
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path=test_helper.TestFiles.polygons_parcels_gpkg
    input1_path = test_helper.prepare_test_file(path, tmp_dir, suffix, crs_epsg)
    path = test_helper.TestFiles.polygons_zones_gpkg
    input2_path = test_helper.prepare_test_file(path, tmp_dir, suffix, crs_epsg)

    ### Test ###
    """
    output_path = tmp_dir / f"{input1_path.stem}_{overlay_operation}_{input2_path.stem}_{discard_nonmatching}{suffix}"
    gfo.overlay(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            spatial_relations_query=spatial_relations_query,
            discard_nonmatching=discard_nonmatching,
            min_area_intersect=min_area_intersect,
            force=True)

    # If no result expected, the output files shouldn't exist
    if expected_featurecount == 0:
        assert output_path.exists() is False
        return

    # Check if the output file is correctly created
    assert output_path.exists() is True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == expected_featurecount
    assert len(layerinfo_output.columns) == (
            len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 1)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None
    """
    input1_gdf = gfo.read_file(input1_path)
    input2_gdf = gfo.read_file(input2_path)

    result_gdf = input1_gdf.overlay(input2_gdf, how=overlay_operation, keep_geom_type=True)
    output_gpd_path = tmp_dir / f"{input1_path.stem}_{overlay_operation}-gpd_{input2_path.stem}_{discard_nonmatching}{suffix}"
    gfo.to_file(result_gdf, output_gpd_path)

def test_select_two_layers(tmpdir):
    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            # If test input file is in wrong format, convert it
            input1_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_parcels_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)

            # If test input file is in wrong format, convert it
            input2_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_zones_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)
        
            # Now run test
            output_path = tmp_dir / f"{input1_path.stem}-output{suffix}"
            print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
            basetest_select_two_layers(input1_path, input2_path, output_path)
    
def basetest_select_two_layers(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Prepare query to execute. At the moment this is just the query for the 
    # intersection() operation.
    input1_layer_info = gfo.get_layerinfo(input1_path)
    input2_layer_info = gfo.get_layerinfo(input2_path)
    primitivetype_to_extract = PrimitiveType(min(
            input1_layer_info.geometrytype.to_primitivetype.value, 
            input2_layer_info.geometrytype.to_primitivetype.value))
    sql_stmt = f'''
            SELECT ST_CollectionExtract(
                    ST_Intersection(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}), 
                    {primitivetype_to_extract.value}) as geom
                    {{layer1_columns_prefix_alias_str}}
                    {{layer2_columns_prefix_alias_str}}
                    ,CASE 
                        WHEN layer2.naam = 'zone1' THEN 'in_zone1'
                        ELSE 'niet_in_zone1'
                        END AS category
                FROM {{input1_databasename}}."{{input1_layer}}" layer1
                JOIN {{input1_databasename}}."rtree_{{input1_layer}}_{{input1_geometrycolumn}}" layer1tree ON layer1.fid = layer1tree.id
                JOIN {{input2_databasename}}."{{input2_layer}}" layer2
                JOIN {{input2_databasename}}."rtree_{{input2_layer}}_{{input2_geometrycolumn}}" layer2tree ON layer2.fid = layer2tree.id
            WHERE 1=1
                {{batch_filter}}
                AND layer1tree.minx <= layer2tree.maxx AND layer1tree.maxx >= layer2tree.minx
                AND layer1tree.miny <= layer2tree.maxy AND layer1tree.maxy >= layer2tree.miny
                AND ST_Intersects(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 1
                AND ST_Touches(layer1.{{input1_geometrycolumn}}, layer2.{{input2_geometrycolumn}}) = 0
            '''
    gfo.select_two_layers(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 29
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 1) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_split(tmpdir):
    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            # If test input file is in wrong format, convert it
            input1_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_parcels_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)

            # If test input file is in wrong format, convert it
            input2_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_zones_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)
        
            # Now run test
            output_path = tmp_dir / f"{input1_path.stem}-output{suffix}"
            print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
            basetest_split_layers(input1_path, input2_path, output_path)
    
def basetest_split_layers(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Do operation
    gfo.split(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 66
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_union(tmpdir):
    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            # If test input file is in wrong format, convert it
            input1_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_parcels_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)

            # If test input file is in wrong format, convert it
            input2_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_zones_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)
        
            # Now run test
            output_path = tmp_dir / f"{input1_path.stem}-output{suffix}"
            print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
            basetest_union(input1_path, input2_path, output_path)
    
def basetest_union(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Do operation
    gfo.union(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            verbose=True)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 71
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_union_circles(tmpdir):

    # Prepare test data + run tests
    tmp_dir = Path(tmpdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for suffix in test_helper.get_test_suffix_list():
        for crs_epsg in test_helper.get_test_crs_epsg_list():
            # If test input file is in wrong format, convert it
            input1_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_overlappingcircles_one_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)

            # If test input file is in wrong format, convert it
            input2_path = test_helper.prepare_test_file(
                    input_path=test_helper.TestFiles.polygons_overlappingcircles_twothree_gpkg,
                    output_dir=tmp_dir,
                    suffix=suffix,
                    crs_epsg=crs_epsg)
        
            # Now run test
            output_path = tmp_dir / f"{input1_path.stem}-output{suffix}"
            print(f"Run test for suffix {suffix}, crs_epsg {crs_epsg}")
            basetest_union_circles(tmp_dir, input1_path, input2_path, output_path)
    
def basetest_union_circles(
        tmp_dir: Path,
        input1_path: Path,
        input2_path: Path,
        output_path: Path):
    
    ##### Also run some tests on basic data with circles #####
    ### Union the single circle towards the 2 circles ###
    gfo.union( 
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            verbose=True)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 5
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Union the two circles towards the single circle ###
    input1_path = test_helper.TestFiles.polygons_overlappingcircles_twothree_gpkg
    input2_path = test_helper.TestFiles.polygons_overlappingcircles_one_gpkg
    output_path = Path(tmp_dir) / f"{input1_path.stem}_union_{input2_path.stem}.gpkg"
    gfo.union( 
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            verbose=True)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = gfo.get_layerinfo(input1_path)
    layerinfo_input2 = gfo.get_layerinfo(input2_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 5
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns)) == len(layerinfo_output.columns)

    # Check geometry type
    if output_path.suffix.lower() == '.shp':
        # For shapefiles the type stays POLYGON anyway 
        assert layerinfo_output.geometrytype == GeometryType.POLYGON 
    elif output_path.suffix.lower() == '.gpkg':
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf['geometry'][0] is not None
