# -*- coding: utf-8 -*-
"""
Module with helper functions to expand on some features of geopandas.
"""

import filecmp
import io
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Any, List, Union

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\gdal"
    os.environ["PROJ_LIB"] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\proj"

import fiona
import geopandas as gpd
from osgeo import gdal

from .util import io_util
from .util import ogr_util
from .util import ogr_util_direct

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
gdal.UseExceptions()        # Enable exceptions

# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def getfileinfo(
        path: Union[str, 'os.PathLike[Any]'],
        verbose: bool = False) -> dict:
    """
    Get information about the geofile.

    Args:
        path (PathLike): path to the file to get info about
        verbose (bool, optional): True to enable verbose logging. Defaults to False.

    Returns:
        dict: the information about the file
    """
    # Get info
    path_p = Path(path)
    info_str = ogr_util.vector_info(
            path=path_p, 
            readonly=True,
            verbose=verbose)

    # Prepare result
    result_dict = {}
    result_dict['info_str'] = info_str
    result_dict['layers'] = []
    info_strio = io.StringIO(info_str)
    for line in info_strio.readlines():
        line = line.strip()
        if re.match(r"\A\d: ", line):
            # It is a layer, now extract only the layer name
            logger.debug(f"This is a layer: {line}")
            layername_with_geomtype = re.sub(r"\A\d: ", "", line)
            layername = re.sub(r" [(][a-zA-Z ]+[)]\Z", "", layername_with_geomtype)
            result_dict['layers'].append(layername)

    return result_dict

def getlayerinfo(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        verbose: bool = False) -> dict:
    """
    Get information about a layer in the geofile.

    Args:
        path (PathLike): path to the file to get info about
        layer (str): the layer you want info about. Doesn't need to be 
            specified if there is only one layer in the geofile. 
        verbose (bool, optional): True to enable verbose logging. Defaults to False.

    Returns:
        dict: the information about the layer
    """        
    ##### Init #####
    path_p = Path(path)
    datasource = gdal.OpenEx(path_p)
    if layer is not None:
        datasource_layer = datasource.GetLayer(layer)
    elif datasource.GetLayerCount() == 1:
        datasource_layer = datasource.GetLayerByIndex(0)
    else:
        raise Exception(f"No layer specified, and file has <> 1 layer: {path_p}")
    
    # Prepare result
    result_dict = {}
    result_dict['featurecount'] = datasource_layer.GetFeatureCount()
    result_dict['geometry_column'] = datasource_layer.GetGeometryColumn()

    # Get column names
    columns = []
    layer_defn = datasource_layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        columns.append(layer_defn.GetFieldDefn(i).GetName())
    result_dict['columns'] = columns
    '''
    ##### Get summary info #####
    info_str = ogr_util.vector_info(
            path=path, 
            layer=layer,
            readonly=True,
            report_summary=True,
            verbose=verbose)

    # Prepare result
    result_dict = {}
    result_dict['info_str'] = info_str
    info_strio = StringIO(info_str)
    for line in info_strio.readlines():
        if line.startswith("Feature Count: "):
            line_value = line.strip().replace("Feature Count: ", "")
            result_dict['featurecount'] = int(line_value)
        elif line.startswith("Geometry Column = "):
            line_value = line.strip().replace("Geometry Column = ", "")
            result_dict['geometry_column'] = line_value

    # If no geometry_column info found, check if specific type of file
    if 'geometry_column' not in result_dict:
        _, ext = os.path.splitext(path)
        ext_lower = ext.lower()
        if ext_lower == '.shp':
            result_dict['geometry_column'] = 'geometry'
    
    ##### Get (non geometry) columns #####
    with fiona.open(path) as fio_layer:
        result_dict['columns'] = fio_layer.schema['properties'].keys()
    '''

    return result_dict

def get_only_layer(path: Union[str, 'os.PathLike[Any]']):
    
    path_p = Path(path)
    layers = fiona.listlayers(path_p)
    nb_layers = len(layers)
    if nb_layers == 1:
        return layers[0]
    elif nb_layers == 0:
        raise Exception(f"Error: No layers found in {path_p}")
    else:
        raise Exception(f"Error: More than 1 layer found in {path_p}: {layers}")

def get_default_layer(path: Union[str, 'os.PathLike[Any]']):
    path_p = Path(path)
    return path_p.stem

def create_spatial_index(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        geometry_column: str = 'geom',
        verbose: bool = False):

    # Check input parameters
    path_p = Path(path)
    if layer is None:
        layer = get_only_layer(path_p)

    '''
    sqlite_stmt = f"SELECT CreateSpatialIndex('{layer}', '{geometry_column}')"
    ogr_util.vector_info(path=path, sqlite_stmt=sqlite_stmt)
    '''
    #driver = ogr.GetDriverByName('SQLite')    
    #data_source = driver.CreateDataSource('db.sqlite', ['SPATIALITE=YES'])    
    data_source = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_UPDATE)
    #layer = data_source.CreateLayer('the_table', None, ogr.wkbLineString25D, ['SPATIAL_INDEX=NO'])
    #geometry_column = layer.GetGeometryColumn()
    data_source.ExecuteSQL(f"SELECT CreateSpatialIndex('{layer}', '{geometry_column}')") 

def rename_layer(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str,
        new_layer: str,
        verbose: bool = False):

    # Check input parameters
    path_p = Path(path)
    if layer is None:
        layer = get_only_layer(path_p)

    sqlite_stmt = f'ALTER TABLE "{layer}" RENAME TO "{new_layer}"'
    ogr_util.vector_info(path=path_p, sqlite_stmt=sqlite_stmt)

def add_column(
        path: Union[str, 'os.PathLike[Any]'],
        column_name: str,
        column_type: str = None,
        layer: str = None):

    ##### Init #####
    path_p = Path(path)
    column_name = column_name.lower()
    if layer is None:
        layer = get_only_layer(path_p)
    if column_name not in ('area'):
        raise Exception(f"Unsupported column type: {column_type}")
    if column_type is None:
        if column_name == 'area':
            column_type = 'real'
        else:
            raise Exception(f"Columns type should be specified for colum name: {column_name}")

    ##### Go! #####
    sqlite_stmt = f'ALTER TABLE "{layer}" ADD COLUMN "{column_name}" {column_type}'

    try:
        ogr_util.vector_info(path=path_p, sqlite_stmt=sqlite_stmt, readonly=False)

        if column_name == 'area':
            sqlite_stmt = f'UPDATE "{layer}" SET "{column_name}" = ST_area(geom)'
            ogr_util.vector_info(path=path_p, sqlite_stmt=sqlite_stmt)
    except Exception as ex:
        # If the column exists already, just print warning
        if 'duplicate column name:' in str(ex):
            logger.warning(f"Column {column_name} existed already in {path_p}")
        else:
            raise ex

def read_file(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        columns: List[str] = None,
        bbox = None,
        rows = None) -> gpd.GeoDataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    # Init
    path_p = Path(path)
    ext_lower = path_p.suffix.lower()

    # For file multilayer types, if no layer name specified, check if there is only one layer in the file.
    if(ext_lower in ['.gpkg'] 
       and layer is None):
        listlayers = fiona.listlayers(path_p)
        if len(listlayers) == 1:
            layer = listlayers[0]
        else:
            raise Exception(f"File contains {len(listlayers)} layers: {listlayers}, but layer is not specified: {path}")

    # Depending on the extension... different implementations
    if ext_lower == '.shp':
        return gpd.read_file(str(path_p), bbox=bbox, rows=rows)
    elif ext_lower == '.geojson':
        return gpd.read_file(str(path_p), bbox=bbox, rows=rows)
    elif ext_lower == '.gpkg':
        return gpd.read_file(str(path_p), layer=layer, bbox=bbox, rows=rows)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

def to_file(
        gdf: gpd.GeoDataFrame,
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        index: bool = True):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support adding optional parameter and pass them to next 
    # function, example encoding, float_format,...
    """
    # Check input parameters
    path_p = Path(path)

    # If no layer name specified, use the filename (without extension)
    if layer is None:
        layer = Path(path_p).stem
    # If the dataframe is empty, log warning and return
    if len(gdf) <= 0:
        #logger.warn(f"Cannot write an empty dataframe to {filepath}.{layer}")
        return

    ext_lower = path_p.suffix.lower()
    if ext_lower == '.shp':
        if index is True:
            gdf = gdf.reset_index(inplace=False)
        gdf.to_file(str(path_p))
    elif ext_lower == '.gpkg':
        gdf.to_file(str(path_p), layer=layer, driver="GPKG")
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
        
def get_crs(path: Union[str, 'os.PathLike[Any]']):
    with fiona.open(path, 'r') as geofile:
        return geofile.crs

def is_geofile(path: Union[str, 'os.PathLike[Any]']) -> bool:
    """
    Determines based on the filepath if this is a geofile.
    """
    # Check input parameters
    path_p = Path(path)

    return is_geofile_ext(path_p.suffix)

def is_geofile_ext(file_ext: str) -> bool:
    """
    Determines based on the file extension if this is a geofile.
    """
    file_ext_lower = file_ext.lower()
    if file_ext_lower in ('.shp', '.gpkg', '.geojson'):
        return True
    else:
        return False

def cmp(
        path1: Union[str, 'os.PathLike[Any]'], 
        path2: Union[str, 'os.PathLike[Any]']) -> bool:
    """
    Compare if two geofiles are identical. 

    For geofiles that use multiple files, all relevant files must be identical.
    Eg. for shapefiles, the .shp, .shx and .dbf file must be identical.

    Args:
        path1 (PathLike): path to the first file
        path2 (PathLike): path to the second file

    Returns:
        bool: True if the files are identical
    """
    # Check input parameters
    path1_p = Path(path1)
    path2_p = Path(path2)

    # For a shapefile, multiple files need to be compared
    if path1_p.suffix.lower() == '.shp':
        path2_noext, _ = os.path.splitext(path2_p)
        shapefile_extentions = [".shp", ".dbf", ".shx"]
        path1_noext = path1_p.parent / path1_p.stem
        path2_noext = path2_p.parent / path2_p.stem
        for ext in shapefile_extentions:
            if not filecmp.cmp(str(path1_noext) + ext, str(path2_noext) + ext):
                logger.info(f"File {path1_noext}{ext} is different from {path2_noext}{ext}")
                return False
        return True
    else:
        return filecmp.cmp(str(path1_p), str(path2_p))
    
def copy(
        src: Union[str, 'os.PathLike[Any]'], 
        dst: Union[str, 'os.PathLike[Any]']):
    """
    Moves the geofile from src to dst. Is the source file is a geofile containing
    of multiple files (eg. .shp) all files are moved.

    Args:
        src (PathLike): the file to move
        dst (PathLike): the location to move the file(s) to
    """
    # Check input parameters
    src_p = Path(src)
    dst_p = Path(dst)

    # For a shapefile, multiple files need to be copied
    if src_p.suffix.lower() == '.shp':
        shapefile_extentions = [".shp", ".dbf", ".shx", ".prj"]

        # If dest is a dir, just use copy. Otherwise concat dest filepaths
        src_noext = src_p.parent / src_p.stem
        if dst_p.is_dir():
            for ext in shapefile_extentions:
                shutil.copy(str(src_noext) + ext, dst_p)
        else:
            dst_noext = dst_p.parent / dst_p.stem
            for ext in shapefile_extentions:
                shutil.copy(str(src_noext) + ext, str(dst_noext) + ext)                
    else:
        return shutil.copy(src_p, dst_p)

def move(
        src: Union[str, 'os.PathLike[Any]'], 
        dst: Union[str, 'os.PathLike[Any]']):
    """
    Moves the geofile from src to dst. Is the source file is a geofile containing
    of multiple files (eg. .shp) all files are moved.

    Args:
        src (PathLike): the file to move
        dst (PathLike): the location to move the file(s) to
    """
    # Check input parameters
    src_p = Path(src)
    dst_p = Path(dst)

    # For a shapefile, multiple files need to be copied
    if src_p.suffix.lower() == '.shp':
        shapefile_extentions = [".shp", ".dbf", ".shx", ".prj"]

        # If dest is a dir, just use move. Otherwise concat dest filepaths
        src_noext = src_p.parent / src_p.stem
        if dst_p.is_dir():
            for ext in shapefile_extentions:
                shutil.move(str(src_noext) + ext, dst_p)
        else:
            dst_noext = dst_p.parent / dst_p.stem
            for ext in shapefile_extentions:
                shutil.move(str(src_noext) + ext, str(dst_noext) + ext)                
    else:
        return shutil.move(str(src_p), dst_p, copy_function=io_util.copyfile)

def get_driver(path: Union[str, 'os.PathLike[Any]']) -> str:
    """
    Get the driver to use for the file extension of this filepath.
    """
    path_p = Path(path)
    return get_driver_for_ext(path_p.suffix)

def get_driver_for_ext(file_ext: str) -> str:
    """
    Get the driver to use for this file extension.
    """
    file_ext_lower = file_ext.lower()
    if file_ext_lower == '.shp':
        return 'ESRI Shapefile'
    elif file_ext_lower == '.geojson':
        return 'GeoJSON'
    elif file_ext_lower == '.gpkg':
        return 'GPKG'
    else:
        raise Exception(f"Not implemented for extension {file_ext_lower}")        
