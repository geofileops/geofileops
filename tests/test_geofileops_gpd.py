# -*- coding: utf-8 -*-
"""
Tests for operations using GeoPandas.
"""

from pathlib import Path
import sys

import geopandas as gpd

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.geofile import GeometryType
from geofileops.util import geofileops_gpd
from geofileops.util.general_util import ParallelizationConfig
from geofileops.util.geometry_util import SimplifyAlgorithm
from tests import test_helper

def get_nb_parallel() -> int:
    # The number of parallel processes to use for these tests.
    return 2

def get_parallelization_config() -> ParallelizationConfig:
    #default_config = ParallelizationConfig()
    test_config = ParallelizationConfig(
            #bytes_basefootprint: int = 50*1024*1024, 
            #bytes_per_row: int = 100, 
            min_avg_rows_per_batch=1, 
            max_avg_rows_per_batch=5, 
            #bytes_min_per_process=None, 
            #bytes_usable=None
            )
    return test_config

def test_buffer_gpkg(tmpdir):
    # Buffer polygon source to test dir
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_buffer(input_path, output_path, GeometryType.MULTIPOLYGON)

    # Buffer point source to test dir
    input_path = test_helper.TestFiles.points_gpkg
    output_path = Path(tmpdir) / 'points-output.gpkg'
    basetest_buffer(input_path, output_path, GeometryType.MULTIPOINT)

    # Buffer line source to test dir
    input_path = test_helper.TestFiles.linestrings_rows_of_trees_gpkg
    output_path = Path(tmpdir) / 'linestrings_rows_of_trees-output.gpkg'
    basetest_buffer(input_path, output_path, GeometryType.MULTILINESTRING)

def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_buffer(input_path, output_path, GeometryType.MULTIPOLYGON)

def basetest_buffer(
        input_path: Path, 
        output_path: Path, 
        input_geometry_type: GeometryType):
    layerinfo_input = geofile.get_layerinfo(input_path)
    
    ### Test positive buffer ###
    geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=1,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_input.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_output.columns) == len(layerinfo_input.columns)
    
    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Read result for some more detailed checks
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Test negative buffer ###
    output_path = output_path.parent / f"{output_path.stem}_m10m{output_path.suffix}"
    geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=-10,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    if input_geometry_type in [GeometryType.MULTIPOINT, GeometryType.MULTILINESTRING]:
        # A Negative buffer of points or linestrings doesn't give a result.
        assert output_path.exists() == False
    else:    
        # A Negative buffer of polygons gives a result for large polygons.
        assert output_path.exists() == True
        layerinfo_output = geofile.get_layerinfo(output_path)
        assert len(layerinfo_output.columns) == len(layerinfo_input.columns) 
        # 7 polygons disappear because of the negative buffer
        assert layerinfo_output.featurecount == layerinfo_input.featurecount - 7
        
        # Check geometry type
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

        # Read result for some more detailed checks
        output_gdf = geofile.read_file(output_path)
        assert output_gdf['geometry'][0] is not None
    
    ### Test negative buffer with explodecollections ###
    output_path = output_path.parent / f"{output_path.stem}_m10m_explode{output_path.suffix}"
    geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=-10,
            explodecollections=True,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    if input_geometry_type in [GeometryType.MULTIPOINT, GeometryType.MULTILINESTRING]:
        # A Negative buffer of points or linestrings doesn't give a result.
        assert output_path.exists() == False
    else:    
        # A Negative buffer of polygons gives a result for large polygons
        assert output_path.exists() == True
        layerinfo_output = geofile.get_layerinfo(output_path)
        assert len(layerinfo_output.columns) == len(layerinfo_input.columns) 
        # 6 polygons disappear because of the negative buffer, 3 polygons are 
        # split in 2 because of the negative buffer and/or explodecollections=True.
        assert layerinfo_output.featurecount == layerinfo_input.featurecount - 7 + 3
        
        # Check geometry type
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

        # Read result for some more detailed checks
        output_gdf = geofile.read_file(output_path)
        assert output_gdf['geometry'][0] is not None

def test_buffer_various_options_gpkg(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_buffer_various_options(input_path, output_path)

def test_buffer_various_options_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_buffer_various_options(input_path, output_path)

def basetest_buffer_various_options(input_path, output_path):

    ### Check if columns parameter works (case insensitive) ###
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    geofileops_gpd.buffer(
            input_path=input_path,
            columns=columns,
            output_path=output_path,
            distance=1,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert 'OIDN' in layerinfo_output.columns
    assert 'UIDN' in layerinfo_output.columns
    assert len(layerinfo_output.columns) == len(columns)

    # Read result for some more detailed checks
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

    ### Check if ... parameter works ###
    # TODO: increase test coverage of other options...

def test_convexhull_gpkg(tmpdir):
    # Select some data from input to output file
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_convexhull(input_path, output_path)

def test_convexhull_shp(tmpdir):
    # Select some data from input to output file
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_convexhull(input_path, output_path)

def basetest_convexhull(input_path, output_path):
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.convexhull(
            input_path=input_path,
            output_path=output_path,
            nb_parallel=get_nb_parallel())

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Read result for some more detailed checks
    output_gdf = geofile.read_file(output_path)
    assert output_gdf['geometry'][0] is not None

def test_dissolve_linestrings_nogroupby_gpkg(tmpdir):
    # Apply operation
    input_path = test_helper.TestFiles.linestrings_watercourses_gpkg
    output_path = Path(tmpdir) / 'linestrings_watercourses-output.gpkg'
    basetest_dissolve_linestrings_nogroupby(input_path, output_path)

def basetest_dissolve_linestrings_nogroupby(input_path, output_basepath):
    # Apply dissolve with explodecollections
    output_path = (output_basepath.parent / 
            f"{output_basepath.stem}_expl{output_basepath.suffix}")
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            explodecollections=True,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config())

    # Check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_path)
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 85
    assert layerinfo_output.geometrytype is GeometryType.LINESTRING
    assert len(layerinfo_output.columns) >= 0

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    # Apply dissolve without explodecollections
    output_path = (output_basepath.parent / 
            f"{output_basepath.stem}_noexpl{output_basepath.suffix}")
    # explodecollections=False only supported if 
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            explodecollections=False,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config())

    # Check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.get_layerinfo(input_path)
    
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 1
    assert layerinfo_output.geometrytype is layerinfo_orig.geometrytype
    assert len(layerinfo_output.columns) >= 0

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

def test_dissolve_polygons_groupby_gpkg(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_dissolve_polygons_groupby(input_path, output_path)

def test_dissolve_polygons_groupby_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_dissolve_polygons_groupby(input_path, output_path)

def basetest_dissolve_polygons_groupby(
        input_path: Path, 
        output_basepath: Path):
    # Init
    layerinfo_input = geofile.get_layerinfo(input_path)

    ### Test dissolve polygons with groupby + without explodecollections ###
    output_path = output_basepath.parent / f"{output_basepath.stem}_group{output_basepath.suffix}"
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=['GEWASGROEP'],
            explodecollections=False,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 6
    assert len(layerinfo_output.columns) == 1

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    ### Test dissolve polygons with explodecollections ###
    output_path = output_basepath.parent / f"{output_basepath.stem}_group_explode{output_basepath.suffix}"
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=['GEWASGROEP'],
            explodecollections=True,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 25
    assert len(layerinfo_output.columns) == 1

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    ### Test dissolve polygons with explodecollections + all columns ###
    output_path = output_basepath.parent / f"{output_basepath.stem}_group_explode_allcolumns{output_basepath.suffix}"
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=['GEWASGROEP'],
            columns=None,
            explodecollections=True,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 25
    assert len(layerinfo_output.columns) == len(layerinfo_input.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    ### Test dissolve polygons with specified output layer ###
    # A different output layer is not supported for shapefile!!!
    try:
        output_path = output_basepath.parent / f"{output_basepath.stem}_group_outputlayer{output_basepath.suffix}"
        geofileops_gpd.dissolve(
                input_path=input_path,
                output_path=output_path,
                groupby_columns=['GEWASGROEP'],
                output_layer='banana',
                nb_parallel=get_nb_parallel(),
                parallelization_config=get_parallelization_config())
    except Exception as ex:
        # A different output_layer is not supported for shapefile, so normal 
        # that an exception is thrown!
        assert output_path.suffix.lower() == '.shp'

    # Now check if the tmp file is correctly created
    if output_path.suffix.lower() != '.shp':
        assert output_path.exists() == True
        layerinfo_output = geofile.get_layerinfo(output_path)
        assert layerinfo_output.featurecount == 25
        assert len(layerinfo_output.columns) == 1
        assert layerinfo_output.name == 'banana'

        # Check geometry type
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

        # Now check the contents of the result file
        input_gdf = geofile.read_file(input_path)
        output_gdf = geofile.read_file(output_path)
        assert input_gdf.crs == output_gdf.crs
        assert len(output_gdf) == layerinfo_output.featurecount
        assert output_gdf['geometry'][0] is not None

def test_dissolve_polygons_nogroupby_gpkg(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_basepath = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_dissolve_polygons_nogroupby(input_path, output_basepath)

def test_dissolve_polygons_nogroupby_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_basepath = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_dissolve_polygons_nogroupby(input_path, output_basepath)

def basetest_dissolve_polygons_nogroupby(
        input_path: Path, 
        output_basepath: Path):
    # Init
    layerinfo_input = geofile.get_layerinfo(input_path)
    
    ### Test dissolve polygons with explodecollections=True (= default) ###
    output_path = output_basepath.parent / f"{output_basepath.stem}_defaults{output_basepath.suffix}"
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config(),
            force=True)

    # Now check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 23
    if output_basepath.suffix == '.shp':
        # Shapefile always has an FID field
        # TODO: think about whether this should also be the case for geopackage??? 
        assert len(layerinfo_output.columns) == 1
    else:
        assert len(layerinfo_output.columns) == 0

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    ### Test dissolve polygons with explodecollections=False ###
    output_path = output_basepath.parent / f"{output_basepath.stem}_defaults{output_basepath.suffix}"
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            explodecollections=False,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config(),
            force=True)

    # Now check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 1
    if output_basepath.suffix == '.shp':
        # Shapefile always has an FID field
        # TODO: think about whether this should also be the case for geopackage??? 
        assert len(layerinfo_output.columns) == 1
    else:
        assert len(layerinfo_output.columns) == 0

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    ### Test dissolve polygons, with output_layer ###
    # A different output layer is not supported for shapefile!!!
    try:
        output_path = output_basepath.parent / f"{output_basepath.stem}_outputlayer{output_basepath.suffix}"
        geofileops_gpd.dissolve(
                input_path=input_path,
                output_path=output_path,
                output_layer='banana',
                explodecollections=True,
                nb_parallel=get_nb_parallel(),
                parallelization_config=get_parallelization_config(),
                force=True)
    except Exception as ex:
        # A different output_layer is not supported for shapefile, so normal 
        # that an exception is thrown!
        assert output_path.suffix.lower() == '.shp'

    # Now check if the result file is correctly created
    if output_path.suffix.lower() != '.shp':
        assert output_path.exists() == True
        layerinfo_output = geofile.get_layerinfo(output_path)
        assert layerinfo_output.featurecount == 23
        assert len(layerinfo_output.columns) == 0
        if output_basepath.suffix == '.shp':
            # Shapefile doesn't support specifying an output_layer
            assert layerinfo_output.name == output_path.stem
        else:
            assert layerinfo_output.name == 'banana'

        # Check geometry type
        assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

        # Now check the contents of the result file
        input_gdf = geofile.read_file(input_path)
        output_gdf = geofile.read_file(output_path)
        assert input_gdf.crs == output_gdf.crs
        assert len(output_gdf) == layerinfo_output.featurecount
        assert output_gdf['geometry'][0] is not None

def test_dissolve_multisinglepolygons_gpkg(tmpdir):
    # Test to check if it is handled well that a file that results in single 
    # and multipolygons during dissolve is treated correctly, as geopackage 
    # doesn't support single and multi-polygons in one layer.
    
    # Init
    tmpdir = Path(tmpdir)
    
    # Create test data
    input_gdf = gpd.GeoDataFrame(geometry=[test_helper.TestData.polygon, test_helper.TestData.multipolygon])
    input_path = tmpdir / 'test_polygon_input.gpkg'
    geofile.to_file(input_gdf, input_path)
    output_path = tmpdir / f"{input_path.stem}_diss.gpkg"
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            explodecollections=True,
            nb_squarish_tiles=2,
            nb_parallel=get_nb_parallel(),
            parallelization_config=get_parallelization_config(),
            force=True)

    # Now check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 3
    assert len(layerinfo_output.columns) == 0

    # Check geometry type
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON 

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

def test_simplify_gpkg(tmpdir):
    # Simplify polygon source to test dir
    input_path = test_helper.TestFiles.polygons_parcels_gpkg
    output_path = Path(tmpdir) / 'polygons_parcels-output.gpkg'
    basetest_simplify(input_path, output_path, GeometryType.MULTIPOLYGON)

    # Simplify point source to test dir
    input_path = test_helper.TestFiles.points_gpkg
    output_path = Path(tmpdir) / 'points-output.gpkg'
    basetest_simplify(input_path, output_path, GeometryType.MULTIPOINT)

    # Simplify line source to test dir
    input_path = test_helper.TestFiles.linestrings_rows_of_trees_gpkg
    output_path = Path(tmpdir) / 'linestrings_rows_of_trees-output.gpkg'
    basetest_simplify(input_path, output_path, GeometryType.MULTILINESTRING)

def test_simplify_shp(tmpdir):
    # Buffer to test dir
    input_path = test_helper.TestFiles.polygons_parcels_shp
    output_path = Path(tmpdir) / 'polygons_parcels-output.shp'
    basetest_simplify(input_path, output_path, GeometryType.MULTIPOLYGON)

def basetest_simplify(
        input_path: Path, 
        output_path: Path, 
        expected_output_geometrytype: GeometryType):

    ### Test default algorithm, rdp ###
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=5,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == expected_output_geometrytype

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    ### Test vw (visvalingam-whyatt) algorithm ###
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=5,
            algorithm=SimplifyAlgorithm.VISVALINGAM_WHYATT,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == expected_output_geometrytype

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

    ### Test lang algorithm ###
    layerinfo_orig = geofile.get_layerinfo(input_path)
    geofileops_gpd.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=5,
            algorithm=SimplifyAlgorithm.LANG,
            lookahead=8,
            nb_parallel=get_nb_parallel())

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.get_layerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_output.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

    # Check geometry type
    assert layerinfo_output.geometrytype == expected_output_geometrytype

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    assert output_gdf['geometry'][0] is not None

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Run
    #test_buffer_gpkg(tmpdir)
    #test_buffer_various_options_gpkg(tmpdir)
    #test_dissolve_linestrings_nogroupby_gpkg(tmpdir)
    #test_dissolve_linestrings_nogroupby_shp(tmpdir)
    test_dissolve_polygons_groupby_gpkg(tmpdir)
    #test_dissolve_polygons_groupby_shp(tmpdir)
    #test_dissolve_polygons_nogroupby_gpkg(tmpdir)
    #test_dissolve_polygons_nogroupby_shp(tmpdir)
    #test_dissolve_multisinglepolygons_gpkg(tmpdir)
    #test_simplify_gpkg(tmpdir)
