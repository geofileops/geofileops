# -*- coding: utf-8 -*-
"""
Tests for operations that are executed using a sql statement on two layers.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.geofile import GeometryType, PrimitiveType
from geofileops.util import geofileops_sql
from geofileops.util.general_util import MissingRuntimeDependencyError
from tests import test_helper

def test_erase_gpkg(tmpdir):
    # Erase from polygon layer
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    erase_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_erase_zones.gpkg'
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON)

    # Erase from point layer
    input_path = test_helper.TestFiles.points_gpkg
    erase_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'points_erase_zones.gpkg'
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOINT)

    # Erase from line layer
    input_path = test_helper.TestFiles.linestrings_rows_of_trees_gpkg
    erase_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'rows_of_trees_erase_zones.gpkg'
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTILINESTRING)

def test_erase_shp(tmpdir):
    # Prepare input and output paths
    input_path = test_helper.TestFiles.polygons_parcels_shp
    erase_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_erase_zones.shp'
    basetest_erase(input_path, erase_path, output_path, 
            expected_output_geometrytype=GeometryType.MULTIPOLYGON)

def basetest_erase(
        input_path: Path,
        erase_path: Path, 
        output_path: Path,
        expected_output_geometrytype: GeometryType):

    # Do operation
    geofileops_sql.erase(
            input_path=input_path, erase_path=erase_path,
            output_path=output_path)

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
    input_to_select_from_path = test_helper.TestFiles.polygons_parcels_gpkg
    input_to_compare_with_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_loc_zones.gpkg'
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path)
        
def test_export_by_location_shp(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.TestFiles.polygons_parcels_shp
    input_to_compare_with_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_loc_zones.shp'
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path)

def basetest_export_by_location(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path):

    geofileops_sql.export_by_location(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path)

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
    input_to_select_from_path = test_helper.TestFiles.polygons_parcels_gpkg
    input_to_compare_with_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_distance_zones.gpkg'
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path)
    
def test_export_by_distance_shp(tmpdir):
    # Prepare input and output paths
    input_to_select_from_path = test_helper.TestFiles.polygons_parcels_shp
    input_to_compare_with_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_distance_zones.shp'
    basetest_export_by_location(
            input_to_select_from_path, input_to_compare_with_path, output_path)

def basetest_export_by_distance(
        input_to_select_from_path: Path, 
        input_to_compare_with_path: Path, 
        output_path: Path):

    geofileops_sql.export_by_distance(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            max_distance=10,
            output_path=output_path)

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
    input1_path = test_helper.TestFiles.polygons_parcels_gpkg
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_intersect_zones.gpkg'
    basetest_intersect(input1_path, input2_path, output_path)
    
def test_intersect_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_shp
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_intersect_zones.gpkg'
    basetest_intersect(input1_path, input2_path, output_path)
    
def basetest_intersect(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Do operation
    geofileops_sql.intersect(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            nb_parallel=2)

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
    input1_path = test_helper.TestFiles.polygons_parcels_gpkg
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_join_zones.gpkg'
    basetest_join_by_location(input1_path, input2_path, output_path)
    
def test_join_by_location_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_shp
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_join_zones.gpkg'
    basetest_join_by_location(input1_path, input2_path, output_path)
    
def basetest_join_by_location(
        input1_path: Path, input2_path: Path,
        output_path: Path):
        
    ### Test 1: inner join, intersect
    geofileops_sql.join_by_location(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            discard_nonmatching=True,
            force=True)

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
    geofileops_sql.join_by_location(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            discard_nonmatching=False,
            force=True)

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

def test_join_nearest_gpkg(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_gpkg
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_nearest_zones.gpkg'
    basetest_join_nearest(input1_path, input2_path, output_path)
    
def test_join_nearest_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_shp
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_nearest_zones.gpkg'
    basetest_join_nearest(input1_path, input2_path, output_path)
    
def basetest_join_nearest(
        input1_path: Path, 
        input2_path: Path,
        output_path: Path):
        
    ### Test 1: inner join, intersect
    nb_nearest = 2
    geofileops_sql.join_nearest(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            nb_nearest=nb_nearest,
            force=True)

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == nb_nearest * layerinfo_input1.featurecount
    assert (len(layerinfo_input1.columns) + len(layerinfo_input2.columns) + 2) == len(layerinfo_output.columns)

    # Check geometry type
    if output_path.suffix.lower() == '.shp':
        # For shapefiles the type stays POLYGON anyway 
        assert layerinfo_output.geometrytype == GeometryType.POLYGON 
    elif output_path.suffix.lower() == '.gpkg':
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_select_two_layers_gpkg(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_gpkg
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_select_zones.gpkg'
    basetest_select_two_layers(input1_path, input2_path, output_path)
    
def test_select_two_layers_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_shp
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_select_zones.gpkg'
    basetest_select_two_layers(input1_path, input2_path, output_path)
    
def basetest_select_two_layers(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Prepare query to execute. At the moment this is just the query for the 
    # intersect() operation.
    input1_layer_info = geofile.get_layerinfo(input1_path)
    input2_layer_info = geofile.get_layerinfo(input2_path)
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
    geofileops_sql.select_two_layers(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            sql_stmt=sql_stmt)

    # Now check if the tmp file is correctly created
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

def test_split_gpkg(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_gpkg
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_split_zones.gpkg'
    basetest_split_layers(input1_path, input2_path, output_path)
    
def test_split_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_shp
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_split_zones.gpkg'
    basetest_split_layers(input1_path, input2_path, output_path)
    
def basetest_split_layers(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Do operation
    geofileops_sql.split(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 65
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
    ##### Run some tests on parcels versus zones #####
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_gpkg
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_union_zones.gpkg'
    basetest_union(input1_path, input2_path, output_path)
    
    ##### Also run some tests on basic data with circles #####
    ### Union the single circle towards the 2 circles ###
    input1_path = test_helper.TestFiles.polygons_overlappingcircles_one_gpkg
    input2_path = test_helper.TestFiles.polygons_overlappingcircles_twothree_gpkg
    output_path = Path(tmpdir) / f"{input1_path.stem}_union_{input2_path.stem}.gpkg"
    geofileops_sql.union( 
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            verbose=True)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 5
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

    ### Union the two circles towards the single circle ###
    input1_path = test_helper.TestFiles.polygons_overlappingcircles_twothree_gpkg
    input2_path = test_helper.TestFiles.polygons_overlappingcircles_one_gpkg
    output_path = Path(tmpdir) / f"{input1_path.stem}_union_{input2_path.stem}.gpkg"
    geofileops_sql.union( 
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            verbose=True)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 5
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

def test_union_shp(tmpdir):
    # Prepare input and output paths
    input1_path = test_helper.TestFiles.polygons_parcels_shp
    input2_path = test_helper.TestFiles.polygons_zones_gpkg
    output_path = Path(tmpdir) / 'parcels-2020_union_zones.gpkg'
    basetest_union(input1_path, input2_path, output_path)
    
def basetest_union(
        input1_path: Path, 
        input2_path: Path, 
        output_path: Path):

    # Do operation
    geofileops_sql.union(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            verbose=True)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_input1 = geofile.get_layerinfo(input1_path)
    layerinfo_input2 = geofile.get_layerinfo(input2_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 69
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
    #test_erase_shp(tmpdir)
    #test_intersect_gpkg(tmpdir)
    #test_export_by_distance_shp(tmpdir)
    #test_export_by_location_gpkg(tmpdir)
    #test_join_by_location_gpkg(tmpdir)
    #test_select_two_layers_gpkg(tmpdir)
    #test_split_gpkg(tmpdir)
    test_union_gpkg(tmpdir)
    