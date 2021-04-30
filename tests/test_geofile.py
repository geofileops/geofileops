# -*- coding: utf-8 -*-
"""
Tests for functionalities in geofileops.general.
"""

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import shapely.geometry as sh_geom

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofile
from geofileops.util import geoseries_util
from geofileops.util.geometry_util import GeometryType
from tests import test_helper

def test_add_column(tmpdir):
    # First copy test file to tmpdir
    # Now add area column
    src = test_helper.TestFiles.polygons_parcels_gpkg
    tmppath = Path(tmpdir) / src.name
    geofile.copy(src, tmppath)

    # The area column shouldn't be in the test file yet
    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' not in layerinfo.columns
        
    ### Add area column ###
    #with test_helper.GdalBin(gdal_installation='gdal_default'):
    geofile.add_column(tmppath, layer='parcels', name='AREA', type='real', expression='ST_area(geom)')
        
    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' in layerinfo.columns
    
    gdf = geofile.read_file(tmppath)
    assert round(gdf['AREA'].astype('float')[0], 1) == round(gdf['OPPERVL'].astype('float')[0], 1)

    ### Add invalid column type -> should raise an exception
    test_ok = False
    try: 
        geofile.add_column(tmppath, layer='parcels', name='joske', type='joske', expression='ST_area(geom)')
        test_ok = False
    except:
        test_ok = True
    assert test_ok is True

def test_cmp(tmpdir):
    # Copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_shp
    dst = Path(tmpdir) / 'polygons_parcels_output.shp'
    geofile.copy(src, dst)

    # Now compare source and dst file
    assert geofile.cmp(src, dst) == True

def test_convert(tmpdir):
    # Convert polygon test file from shape to geopackage
    src = test_helper.TestFiles.polygons_parcels_shp
    dst = Path(tmpdir) / 'polygons_parcels_output.gpkg'
    geofile.convert(src, dst)

    # Now compare source and dst file 
    src_layerinfo = geofile.get_layerinfo(src)
    dst_layerinfo = geofile.get_layerinfo(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert len(src_layerinfo.columns) == len(dst_layerinfo.columns)

def test_convert_force_output_geometrytype(tmpdir):
    # The conversion is done by ogr, and these test are just written to explore
    # the behaviour of this ogr functionality 
    
    # Convert polygon test file and force to polygon
    src = test_helper.TestFiles.polygons_parcels_gpkg
    dst = Path(tmpdir) / 'polygons_parcels_to_polygon.gpkg'
    geofile.convert(src, dst, force_output_geometrytype=GeometryType.POLYGON)

    # Convert polygon test file and force to multipolygon
    src = test_helper.TestFiles.polygons_parcels_gpkg
    dst = Path(tmpdir) / 'polygons_parcels_to_multipolygon.gpkg'
    geofile.convert(src, dst, force_output_geometrytype=GeometryType.MULTIPOLYGON)

    # Convert polygon test file and force to linestring
    src = test_helper.TestFiles.polygons_parcels_gpkg
    dst = Path(tmpdir) / 'polygons_parcels_to_linestring.gpkg'
    geofile.convert(src, dst, force_output_geometrytype=GeometryType.LINESTRING)

    # Convert polygon test file and force to multilinestring
    src = test_helper.TestFiles.polygons_parcels_gpkg
    dst = Path(tmpdir) / 'polygons_parcels_to_multilinestring.gpkg'
    geofile.convert(src, dst, force_output_geometrytype=GeometryType.MULTILINESTRING)

    # Convert polygon test file and force to point
    src = test_helper.TestFiles.polygons_parcels_gpkg
    dst = Path(tmpdir) / 'polygons_parcels_to_point.gpkg'
    geofile.convert(src, dst, force_output_geometrytype=GeometryType.POINT)

    # Convert polygon test file and force to multipoint
    src = test_helper.TestFiles.polygons_parcels_gpkg
    dst = Path(tmpdir) / 'polygons_parcels_to_point.gpkg'
    geofile.convert(src, dst, force_output_geometrytype=GeometryType.MULTIPOINT)

def test_copy(tmpdir):
    # Copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_shp
    dst = Path(tmpdir) / 'polygons_parcels_output.shp'
    geofile.copy(src, dst)
    assert dst.exists() == True

def test_get_crs():
    # Test shapefile
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    crs = geofile.get_crs(srcpath)
    assert crs.to_epsg() == 31370

    # Test geopackage
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    crs = geofile.get_crs(srcpath)
    assert crs.to_epsg() == 31370

def test_get_default_layer_shp():
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    layer = geofile.get_default_layer(srcpath)
    assert layer == 'polygons_parcels-2020'

def test_get_driver():
    # Test shapefile
    src = test_helper.TestFiles.polygons_parcels_shp
    assert geofile.get_driver(src) == "ESRI Shapefile"

    # Test geopackage
    src = test_helper.TestFiles.polygons_parcels_gpkg
    assert geofile.get_driver(src) == "GPKG"

def test_get_layerinfo():
    
    def basetest_get_layerinfo(srcpath: Path, layer: str = None):
        # Test for file containing one layer
        layerinfo = geofile.get_layerinfo(srcpath, layer)
        assert layerinfo.featurecount == 46
        if srcpath.suffix == '.shp':
            assert layerinfo.geometrycolumn == 'geometry'
            assert layerinfo.name == srcpath.stem
        elif srcpath.suffix == '.gpkg':
            assert layerinfo.geometrycolumn == 'geom'
            assert layerinfo.name == 'parcels'
        assert layerinfo.geometrytypename == geofile.GeometryType.MULTIPOLYGON.name
        assert layerinfo.geometrytype == geofile.GeometryType.MULTIPOLYGON
        assert len(layerinfo.columns) == 10
        assert layerinfo.total_bounds is not None
        assert layerinfo.crs.to_epsg() == 31370

    # Test shapefile
    basetest_get_layerinfo(srcpath=test_helper.TestFiles.polygons_parcels_shp)
    # Test geopackage, one layer
    basetest_get_layerinfo(srcpath=test_helper.TestFiles.polygons_parcels_gpkg)
    # Test geopackage, two layers
    basetest_get_layerinfo(
            srcpath=test_helper.TestFiles.polygons_twolayers_gpkg, 
            layer='parcels')

def test_get_only_layer(tmpdir):
    tmpdir = Path(tmpdir)
    ### Test Geopackage and shapefile with one layer ###
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    layer = geofile.get_only_layer(srcpath)
    assert layer == 'parcels'
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    layer = geofile.get_only_layer(srcpath)
    assert layer == srcpath.stem

    ### Test Geopackage with 2 layers ###
    srcpath = test_helper.TestFiles.polygons_twolayers_gpkg
    layers = geofile.listlayers(srcpath)
    assert len(layers) == 2
    error_raised = False
    try:
        layer = geofile.get_only_layer(srcpath)
    except:
        error_raised = True
    assert error_raised is True

def test_is_geofile():
    assert geofile.is_geofile(test_helper.TestFiles.polygons_parcels_shp) == True
    assert geofile.is_geofile(test_helper.TestFiles.polygons_parcels_gpkg) == True
    
def test_listlayers():
    layers = geofile.listlayers(test_helper.TestFiles.polygons_parcels_shp)
    assert layers[0] == test_helper.TestFiles.polygons_parcels_shp.stem
    layers = geofile.listlayers(test_helper.TestFiles.polygons_parcels_gpkg)
    assert layers[0] == 'parcels'
    layers = geofile.listlayers(test_helper.TestFiles.polygons_twolayers_gpkg)
    assert 'parcels' in layers
    assert 'zones' in layers

def test_move(tmpdir):
    # Copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_shp
    tmp1path = Path(tmpdir) / 'polygons_parcels_tmp.shp'
    geofile.copy(src, tmp1path)
    assert tmp1path.exists() == True

    # Move (rename actually) and check result
    tmp2path = Path(tmpdir) / 'polygons_parcels_tmp2.shp'
    geofile.move(tmp1path, tmp2path)
    assert tmp1path.exists() == False
    assert tmp2path.exists() == True

def test_update_column(tmpdir):
    # First copy test file to tmpdir
    # Now add area column
    src = test_helper.TestFiles.polygons_parcels_gpkg
    tmppath = Path(tmpdir) / 'polygons_parcels.gpkg'
    geofile.copy(src, tmppath)

    # The area column shouldn't be in the test file yet
    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'area' not in layerinfo.columns
        
    ### Add area column ###
    #with test_helper.GdalBin(gdal_installation='gdal_default'):
    geofile.add_column(tmppath, layer='parcels', name='AREA', type='real', expression='ST_area(geom)')
    geofile.update_column(tmppath, name='AreA', expression='ST_area(geom)')

    layerinfo = geofile.get_layerinfo(path=tmppath, layer='parcels')
    assert 'AREA' in layerinfo.columns
    gdf = geofile.read_file(tmppath)
    assert round(gdf['AREA'].astype('float')[0], 1) == round(gdf['OPPERVL'].astype('float')[0], 1)

def test_read_file():
    # Test shapefile
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    basetest_read_file(srcpath=srcpath)

    # Test geopackage
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    basetest_read_file(srcpath=srcpath)

def basetest_read_file(srcpath: Path):
    # Test with defaults
    read_gdf = geofile.read_file(srcpath)
    assert isinstance(read_gdf, gpd.GeoDataFrame)
    assert len(read_gdf) == 46

    # Test no columns
    read_gdf = geofile.read_file(srcpath, columns=[])
    assert isinstance(read_gdf, gpd.GeoDataFrame)
    assert len(read_gdf) == 46

    # Test specific columns (+ test case insensitivity)
    columns = ['OIDN', 'uidn', 'HFDTLT', 'lblhfdtlt', 'GEWASGROEP', 'lengte', 'OPPERVL']
    read_gdf = geofile.read_file(srcpath, columns=columns)
    assert len(read_gdf) == 46
    assert len(read_gdf.columns) == (len(columns) + 1)

    # Test no geom
    read_gdf = geofile.read_file_nogeom(srcpath)
    assert isinstance(read_gdf, pd.DataFrame)
    assert len(read_gdf) == 46

    # Test ignore_geometry, no columns
    read_gdf = geofile.read_file_nogeom(srcpath, columns=[])
    assert isinstance(read_gdf, pd.DataFrame)
    assert len(read_gdf) == 46

def test_rename_layer(tmpdir):
    ### Geopackage ###
    # First copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_gpkg
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.gpkg'
    geofile.copy(src, tmppath)

    # Now test rename layer
    geofile.rename_layer(tmppath, layer='parcels', new_layer='parcels_renamed')
    layernames_renamed = geofile.listlayers(path=tmppath)
    assert layernames_renamed[0] == 'parcels_renamed'

    ### Shapefile ###
    # First copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_shp
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.shp'
    geofile.copy(src, tmppath)

    # Now test rename layer
    try:
        geofile.rename_layer(tmppath, layer='polygons_parcels_tmp', new_layer='polygons_parcels_tmp_renamed')
        layernames_renamed = geofile.listlayers(path=tmppath)
        assert layernames_renamed[0] == 'polygons_parcels_tmp_renamed'
    except Exception as ex:
        assert 'rename_layer is not possible' in str(ex) 

def test_execute_sql(tmpdir):
    # First copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_gpkg
    tmppath = Path(tmpdir) / 'polygons_parcels.gpkg'
    geofile.copy(src, tmppath)

    ### Test using execute_sql for creating/dropping indexes ###
    # Create index
    geofile.execute_sql(path=tmppath, sql_stmt='CREATE INDEX idx_parcels_oidn ON "parcels"("oidn")')

    # Drop index
    geofile.execute_sql(path=tmppath, sql_stmt='DROP INDEX idx_parcels_oidn')

def test_spatial_index_gpkg(tmpdir):
    # First copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_gpkg
    tmppath = Path(tmpdir) / 'polygons_parcels.gpkg'
    geofile.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = geofile.has_spatial_index(
            path=tmppath, layer='parcels')
    assert has_spatial_index is True

    # Remove spatial index
    geofile.remove_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = geofile.has_spatial_index(
            path=tmppath, layer='parcels')
    assert has_spatial_index is False

    # Create spatial index
    geofile.create_spatial_index(path=tmppath, layer='parcels')
    has_spatial_index = geofile.has_spatial_index(
            path=tmppath, layer='parcels')
    assert has_spatial_index is True

def test_spatial_index_shp(tmpdir):
    # First copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_shp
    tmppath = Path(tmpdir) / 'polygons_parcels.shp'
    geofile.copy(src, tmppath)

    # Check if spatial index present
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index is True

    # Remove spatial index
    geofile.remove_spatial_index(path=tmppath)
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index is False

    # Create spatial index
    geofile.create_spatial_index(path=tmppath)
    has_spatial_index = geofile.has_spatial_index(path=tmppath)
    assert has_spatial_index is True

def test_to_file_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.shp'
    basetest_to_file(srcpath, tmppath)

def test_to_file_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.gpkg'
    basetest_to_file(srcpath, tmppath)

def basetest_to_file(srcpath, tmppath):
    # Read test file and write to tmppath
    read_gdf = geofile.read_file(srcpath)
    geofile.to_file(read_gdf, tmppath)
    tmp_gdf = geofile.read_file(tmppath)
    assert len(read_gdf) == len(tmp_gdf)

    # Append the file again to tmppath
    geofile.to_file(read_gdf, tmppath, append=True)
    tmp_gdf = geofile.read_file(tmppath)
    assert 2*len(read_gdf) == len(tmp_gdf)

def test_to_file_empty_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    output_suffix = '.gpkg'
    basetest_to_file_empty(srcpath, Path(tmpdir), output_suffix)

def test_to_file_empty_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    output_suffix = '.shp'
    basetest_to_file_empty(srcpath, Path(tmpdir), output_suffix)

def basetest_to_file_empty(
        srcpath: Path, 
        output_dir: Path,
        output_suffix: str):
    ### Test for gdf with a None geometry + a polygon ###
    test_gdf = gpd.GeoDataFrame(geometry=[
            None, test_helper.TestData.polygon])
    test_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_geometrytypes) == 1
    output_none_path = output_dir / f"{srcpath.stem}_none{output_suffix}"
    geofile.to_file(test_gdf, output_none_path)
    
    # Now check the result if the data is still the same after being read again
    test_read_gdf = geofile.read_file(output_none_path)
    # Result is the same as the original input
    assert test_read_gdf.geometry[0] is None
    assert isinstance(test_read_gdf.geometry[1], sh_geom.Polygon)  
    # The geometrytype of the column in the file is also the same as originaly
    test_file_geometrytype = geofile.get_layerinfo(output_none_path).geometrytype
    if output_suffix == '.shp':
        assert test_file_geometrytype == GeometryType.MULTIPOLYGON
    else:
        assert test_file_geometrytype == test_geometrytypes[0]
    # The result type in the geodataframe is also the same as originaly
    test_read_geometrytypes = geoseries_util.get_geometrytypes(test_read_gdf.geometry)
    assert len(test_gdf) == len(test_read_gdf)
    assert test_read_geometrytypes == test_geometrytypes

def test_to_file_none_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    output_suffix = '.gpkg'
    basetest_to_file_none(srcpath, Path(tmpdir), output_suffix)

def test_to_file_none_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    output_suffix = '.shp'
    basetest_to_file_none(srcpath, Path(tmpdir), output_suffix)

def basetest_to_file_none(
        srcpath: Path, 
        output_dir: Path,
        output_suffix: str):
    ### Test for gdf with a None geometry + a polygon ###
    test_gdf = gpd.GeoDataFrame(geometry=[
            None, test_helper.TestData.polygon])
    test_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_geometrytypes) == 1
    output_none_path = output_dir / f"{srcpath.stem}_none{output_suffix}"
    geofile.to_file(test_gdf, output_none_path)
    
    # Now check the result if the data is still the same after being read again
    test_read_gdf = geofile.read_file(output_none_path)
    # Result is the same as the original input
    assert test_read_gdf.geometry[0] is None
    assert isinstance(test_read_gdf.geometry[1], sh_geom.Polygon)  
    # The geometrytype of the column in the file is also the same as originaly
    test_file_geometrytype = geofile.get_layerinfo(output_none_path).geometrytype
    if output_suffix == '.shp':
        assert test_file_geometrytype == GeometryType.MULTIPOLYGON
    else:
        assert test_file_geometrytype == test_geometrytypes[0]
    # The result type in the geodataframe is also the same as originaly
    test_read_geometrytypes = geoseries_util.get_geometrytypes(test_read_gdf.geometry)
    assert len(test_gdf) == len(test_read_gdf)
    assert test_read_geometrytypes == test_geometrytypes

def test_to_file_gpd_empty_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    output_suffix = '.gpkg'
    basetest_to_file_gpd_empty(srcpath, Path(tmpdir), output_suffix)

def test_to_file_gpd_empty_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    output_suffix = '.shp'
    basetest_to_file_gpd_empty(srcpath, Path(tmpdir), output_suffix)

def basetest_to_file_gpd_empty(
        srcpath: Path, 
        output_dir: Path,
        output_suffix: str):
    ### Test for gdf with an empty polygon + a polygon ###
    test_gdf = gpd.GeoDataFrame(geometry=[
            sh_geom.Polygon(), test_helper.TestData.polygon])
    # By default, get_geometrytypes ignores the type of empty geometries, as 
    # they are always stored as GeometryCollection in GeoPandas
    test_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_geometrytypes) == 1
    test_geometrytypes_includingempty = geoseries_util.get_geometrytypes(
            test_gdf.geometry, ignore_empty_geometries=False)
    assert len(test_geometrytypes_includingempty) == 2
    output_empty_path = output_dir / f"{srcpath.stem}_empty{output_suffix}"
    test_gdf.to_file(output_empty_path, driver=geofile.get_driver_for_ext(output_suffix))
    
    # Now check the result if the data is still the same after being read again
    test_read_gdf = geofile.read_file(output_empty_path)
    test_read_geometrytypes = geoseries_util.get_geometrytypes(test_read_gdf.geometry)
    assert len(test_gdf) == len(test_read_gdf)
    if output_suffix == '.shp':
        # When dataframe with "empty" gemetries is written to shapefile and 
        # read again, shapefile becomes of type MULTILINESTRING!?!
        assert len(test_read_geometrytypes) == 1
        assert test_read_geometrytypes[0] is GeometryType.MULTILINESTRING
    else:
        # When written to Geopackage... the empty geometries are actually saved 
        # as None, so when read again they are None as well.
        assert test_read_gdf.geometry[0] is None
        assert isinstance(test_read_gdf.geometry[1], sh_geom.Polygon)  

        # So the geometrytype of the resulting GeoDataFrame is also POLYGON
        assert len(test_read_geometrytypes) == 1
        assert test_read_geometrytypes[0] is GeometryType.POLYGON

def test_to_file_gpd_none_gpkg(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_gpkg
    output_suffix = '.gpkg'
    basetest_to_file_gpd_none(srcpath, Path(tmpdir), output_suffix)

def test_to_file_gpd_none_shp(tmpdir):
    # Read test file and write to tmpdir
    srcpath = test_helper.TestFiles.polygons_parcels_shp
    output_suffix = '.shp'
    basetest_to_file_gpd_none(srcpath, Path(tmpdir), output_suffix)

def basetest_to_file_gpd_none(
        srcpath: Path, 
        output_dir: Path,
        output_suffix: str):
    ### Test for gdf with a None geometry + a polygon ###
    test_gdf = gpd.GeoDataFrame(geometry=[
            None, test_helper.TestData.polygon])
    test_geometrytypes = geoseries_util.get_geometrytypes(test_gdf.geometry)
    assert len(test_geometrytypes) == 1
    output_none_path = output_dir / f"{srcpath.stem}_none{output_suffix}"
    test_gdf.to_file(output_none_path, driver=geofile.get_driver_for_ext(output_suffix))
    
    # Now check the result if the data is still the same after being read again
    test_read_gdf = geofile.read_file(output_none_path)
    # Result is the same as the original input
    assert test_read_gdf.geometry[0] is None
    assert isinstance(test_read_gdf.geometry[1], sh_geom.Polygon)  
    # The geometrytype of the column in the file is also the same as originaly
    test_file_geometrytype = geofile.get_layerinfo(output_none_path).geometrytype
    if output_suffix == '.shp':
        # Geometrytype of shapefile always returns the multitype  
        assert test_file_geometrytype == test_geometrytypes[0].to_multitype
    else:
        assert test_file_geometrytype == test_geometrytypes[0]
    # The result type in the geodataframe is also the same as originaly
    test_read_geometrytypes = geoseries_util.get_geometrytypes(test_read_gdf.geometry)
    assert len(test_gdf) == len(test_read_gdf)
    assert test_read_geometrytypes == test_geometrytypes

def test_remove(tmpdir):
    # Copy test file to tmpdir
    src = test_helper.TestFiles.polygons_parcels_shp
    tmppath = Path(tmpdir) / 'polygons_parcels_tmp.shp'
    geofile.copy(src, tmppath)
    assert tmppath.exists() == True

    # Remove and check result
    geofile.remove(tmppath)
    assert tmppath.exists() == False

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Run!
    #test_convert(tmpdir)
    #test_convert_force_output_geometrytype(tmpdir)
    #test_get_layerinfo()
    #test_get_only_layer(tmpdir)
    #test_rename_layer(tmpdir)
    #test_listlayers()
    #test_add_column(tmpdir)
    test_execute_sql(tmpdir)
    #test_read_file()
    #test_to_file_gpkg(tmpdir)
    #test_to_file_shp(tmpdir)
    #test_to_file_empty_gpkg(tmpdir)
    #test_to_file_empty_shp(tmpdir)
    #test_to_file_none_gpkg(tmpdir)
    #test_to_file_none_shp(tmpdir)
    
