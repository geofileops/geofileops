# -*- coding: utf-8 -*-
"""
Module with helper functions for geo files.
"""

import datetime
import filecmp
import logging
import os
from pathlib import Path
import pyproj
import shutil
import tempfile
import time
from typing import Any, List, Optional, Tuple, Union

import fiona
import geopandas as gpd
from osgeo import gdal
import pandas as pd

from geofileops.util import geometry_util
from geofileops.util.geometry_util import GeometryType, PrimitiveType
from geofileops.util import geoseries_util
from geofileops.util import io_util
from geofileops.util import ogr_util

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

shapefile_suffixes = ['.shp', '.dbf', '.shx', '.prj', '.qix', '.sbn', '.sbx']
gdal.UseExceptions()        # Enable exceptions

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

'''
def write_vrt(
        input_layer_paths: List[Tuple[Path, str]],
        output_path: Path):
    """
    Create a vrt file where all layers in input_layer_paths.

    Args:
        input_layer_paths (dict): [description]
        output_path (Path): [description]
    """

    # Create vrt file
    with open(output_path, "w") as vrt_file:
        vrt_file.write(f'<OGRVRTDataSource>\n')

        # Loop over all layers and add them to vrt file
        for index, input_layer_path in enumerate(input_layer_paths):
            vrt_file.write(f'  <OGRVRTLayer name="{input_layer_path[1]}">\n')
            vrt_file.write(f'    <SrcDataSource>{input_layer_path[0]}</SrcDataSource>\n')
            vrt_file.write(f'    <SrcLayer>{input_layer_path[1]}</SrcLayer>\n')
            vrt_file.write(f'  </OGRVRTLayer>\n')

            # layer index
            vrt_file.write(f'  <OGRVRTLayer name="rtree_{input_layer_path[1]}_geom">\n')
            vrt_file.write(f'    <SrcDataSource>{input_layer_path[0]}</SrcDataSource>\n')
            vrt_file.write(f'    <SrcLayer>rtree_{input_layer_path[1]}_geom</SrcLayer>\n')
            vrt_file.write(f'    <Field name="minx" />\n')
            vrt_file.write(f'    <Field name="miny" />\n')
            vrt_file.write(f'    <Field name="maxx" />\n')
            vrt_file.write(f'    <Field name="maxy" />\n')
            vrt_file.write(f'    <GeometryType>wkbNone</GeometryType>\n')
            vrt_file.write(f'  </OGRVRTLayer>\n')
            
        vrt_file.write(f'</OGRVRTDataSource>\n')
'''

def listlayers(
        path: Union[str, 'os.PathLike[Any]'],
        verbose: bool = False) -> List[str]:
    """
    Get the list of layers in a geofile.

    Args:
        path (PathLike): path to the file to get info about
        verbose (bool, optional): True to enable verbose logging. Defaults to False.

    Returns:
        List[str]: the list of layers
    """
    return fiona.listlayers(str(path))

class LayerInfo:
    """
    A data object containing meta-information about a layer.

    Attributes:
        name (str): the name of the layer.
        featurecount (int): the number of features (rows) in the layer.
        total_bounds (Tuple[float, float, float, float]): the bounding box of 
            the layer.
        geometrycolumn (str): name of the column that contains the 
            primary geometry.
        geometrytypename (str): the geometry type name of the geometrycolumn. 
            The type name returned is one of the following: POINT, MULTIPOINT, 
            LINESTRING, MULTILINESTRING, POLYGON, MULTIPOLYGON, COLLECTION.
        geometrytype (GeometryType): the geometry type of the geometrycolumn.
        columns (List[str]): the columns (other than the geometry column) that 
            are available on the layer.
        crs (pyproj.CRS): the spatial reference of the layer. 
    """
    def __init__(self, 
            name: str,
            featurecount: int, 
            total_bounds: Tuple[float, float, float, float],
            geometrycolumn: str, 
            geometrytypename: str,
            geometrytype: GeometryType,
            columns: List[str],
            crs: Optional[pyproj.CRS]):
        self.name = name
        self.featurecount = featurecount
        self.total_bounds = total_bounds
        self.geometrycolumn = geometrycolumn
        self.geometrytypename = geometrytypename
        self.geometrytype = geometrytype
        self.columns = columns
        self.crs = crs

    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"

def get_layerinfo(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None) -> LayerInfo:
    """
    Get information about a layer in the geofile.

    Args:
        path (PathLike): path to the file to get info about
        layer (str): the layer you want info about. Doesn't need to be 
            specified if there is only one layer in the geofile. 
        verbose (bool, optional): True to enable verbose logging. Defaults to False.

    Returns:
        LayerInfo: the information about the layer
    """        
    ##### Init #####
    datasource = None
    try:
        datasource = gdal.OpenEx(str(path))
        if layer is not None:
            datasource_layer = datasource.GetLayer(layer)
        elif datasource.GetLayerCount() == 1:
            datasource_layer = datasource.GetLayerByIndex(0)
        else:
            raise Exception(f"No layer specified, and file has <> 1 layer: {path}")

        # If the layer doesn't exist, return 
        if datasource_layer is None:
            raise Exception(f"Layer {layer} not found in file: {path}")

        # Get column info
        columns = []
        layer_defn = datasource_layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            columns.append(layer_defn.GetFieldDefn(i).GetName())

        # Get geometry column info...
        geometrytypename = gdal.ogr.GeometryTypeToName(datasource_layer.GetGeomType())
        geometrytypename = geometrytypename.replace(' ', '').upper()
        
        # For shape files, the difference between the 'MULTI' variant and the 
        # single one doesn't exists... so always report MULTI variant by convention.
        if Path(path).suffix.lower() == '.shp':
            if(geometrytypename.startswith('POLYGON')
            or geometrytypename.startswith('LINESTRING')
            or geometrytypename.startswith('POINT')):
                geometrytypename = f"MULTI{geometrytypename}"
        if geometrytypename == 'UNKNOWN(ANY)':
            geometrytypename = 'GEOMETRY'
            
        # Geometrytype
        if geometrytypename != 'NONE':
            geometrytype = GeometryType[geometrytypename]
        else:
            geometrytype = None

        # If the geometry type is not None, fill out the extra properties    
        geometrycolumn = None
        extent= None
        crs = None
        total_bounds = None
        if geometrytype is not None:
            # Geometry column name
            geometrycolumn = datasource_layer.GetGeometryColumn()
            if geometrycolumn == '':
                geometrycolumn = 'geometry'
            # Convert gdal extent (xmin, xmax, ymin, ymax) to bounds (xmin, ymin, xmax, ymax)
            extent = datasource_layer.GetExtent()
            total_bounds = (extent[0], extent[2], extent[1], extent[3])
            # CRS
            spatialref = datasource_layer.GetSpatialRef()
            if spatialref is not None:
                crs = pyproj.CRS(spatialref.ExportToWkt())

            return LayerInfo(
                    name=datasource_layer.GetName(),
                    featurecount=datasource_layer.GetFeatureCount(),
                    total_bounds=total_bounds,
                    geometrycolumn=geometrycolumn, 
                    geometrytypename=geometrytypename,
                    geometrytype=geometrytype,
                    columns=columns,
                    crs=crs)
        else:
            raise Exception("Layer doesn't have a geometry column!")    

    finally:
        if datasource is not None:
            del datasource

def get_only_layer(path: Union[str, 'os.PathLike[Any]']) -> str:
    """
    Get the layername for a file that only contains one layer.

    If the file contains multiple layers, an exception is thrown.

    Args:
        path (PathLike): the file.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        str: the layer name
    """
    layers = fiona.listlayers(str(path))
    nb_layers = len(layers)
    if nb_layers == 1:
        return layers[0]
    elif nb_layers == 0:
        raise Exception(f"Error: No layers found in {path}")
    else:
        raise Exception(f"Error: More than 1 layer found in {path}: {layers}")

def get_default_layer(path: Union[str, 'os.PathLike[Any]']) -> str:
    """
    Get the default layer name to be used for a layer in this file.

    This is the stem of the filepath.

    Args:
        path (Union[str,): The path to the file.

    Returns:
        str: The default layer name.
    """
    return Path(path).stem

def has_spatial_index(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        geometrycolumn: str = None) -> bool:
    """
    Check if the layer/column has a spatial index.

    Args:
        path (PathLike): The file path.
        layer (str, optional): The layer. Defaults to None.
        geometrycolumn (str, optional): The geometry column to check. 
            Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        bool: True if the spatial column exists.
    """
    # Init
    path_p = Path(path)

    # Now check the index
    driver = get_driver(path_p)    
    if driver == 'GPKG':
        layerinfo = get_layerinfo(path_p, layer)

        datasource = None
        try:
            data_source = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_READONLY)
            result = data_source.ExecuteSQL(
                    f"SELECT HasSpatialIndex('{layerinfo.name}', '{layerinfo.geometrycolumn}')",
                    dialect='SQLITE')
            row = result.GetNextFeature()
            has_spatial_index = row.GetField(0)
            return (has_spatial_index == 1)
        finally:
            if datasource is not None:
                del datasource
    elif driver == 'ESRI Shapefile':
        index_path = path_p.parent / f"{path_p.stem}.qix" 
        return index_path.exists()
    else:
        raise Exception(f"has_spatial_index not supported for {path_p}")

def create_spatial_index(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None):
    """
    Create a spatial index on the layer specified.

    Args:
        path (PathLike): The file path.
        layer (str, optional): The layer. If not specified, and there is only 
            one layer in the file, this layer is used. Otherwise exception.
    """
    # Init
    path_p = Path(path)
    layerinfo = get_layerinfo(path_p, layer)

    # Now really add index
    datasource = None
    try:
        datasource = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_UPDATE)
        driver = get_driver(path_p)    
        if driver == 'GPKG':
            datasource.ExecuteSQL(
                    f"SELECT CreateSpatialIndex('{layerinfo.name}', '{layerinfo.geometrycolumn}')",                
                    dialect='SQLITE') 
        else:
            datasource.ExecuteSQL(f'CREATE SPATIAL INDEX ON "{layerinfo.name}"')
    finally:
        if datasource is not None:
            del datasource

def remove_spatial_index(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None):
    """
    Remove the spatial index from the layer specified.

    Args:
        path (PathLike): The file path.
        layer (str, optional): The layer. If not specified, and there is only 
            one layer in the file, this layer is used. Otherwise exception.
    """
    # Init
    path_p = Path(path)
    layerinfo = get_layerinfo(path_p, layer)

    # Now really remove index
    datasource = None
    try:
        suffix_lower = path_p.suffix.lower()  
        if suffix_lower == '.gpkg':
            datasource = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_UPDATE)
            datasource.ExecuteSQL(
                    f"SELECT DisableSpatialIndex('{layerinfo.name}', '{layerinfo.geometrycolumn}')",
                    dialect='SQLITE') 
        elif suffix_lower == '.shp':
            # DROP SPATIAL INDEX ON ... command gives an error, so just remove .qix
            index_path = path_p.parent / f"{path_p.stem}.qix" 
            index_path.unlink()
        else:
            raise Exception(f"remove_spatial_index is not supported for {suffix_lower} file")
            #datasource = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_UPDATE)
            #datasource.ExecuteSQL(f'DROP SPATIAL INDEX ON "{layerinfo.name}"')
    finally:
        if datasource is not None:
            del datasource

def rename_layer(
        path: Union[str, 'os.PathLike[Any]'],
        new_layer: str,
        layer: Optional[str] = None):
    """
    Rename the layer specified.

    Args:
        path (PathLike): The file path.
        layer (Optional[str]): The layer name. If not specified, and there is only 
            one layer in the file, this layer is used. Otherwise exception.
        new_layer (str): The new layer name. If not specified, and there is only 
            one layer in the file, this layer is used. Otherwise exception.
    """
    # Check input parameters
    path_p = Path(path)
    if layer is None:
        layer = get_only_layer(path_p)

    # Now really rename
    datasource = None
    try:
        suffix_lower = path_p.suffix.lower()  
        if suffix_lower == '.gpkg':
            datasource = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_UPDATE)
            sql_stmt = f'ALTER TABLE "{layer}" RENAME TO "{new_layer}"'
            datasource.ExecuteSQL(sql_stmt)
        elif suffix_lower == '.shp':
            raise Exception(f"rename_layer is not possible for {suffix_lower} file")
        else:
            raise Exception(f"rename_layer is not implemented for {suffix_lower} file")
    finally:
        if datasource is not None:
            del datasource

def add_column(
        path: Union[str, 'os.PathLike[Any]'],
        name: str,
        type: str,
        expression: Union[str, int, float] = None, 
        layer: str = None,
        force_update: bool = False):
    """
    Add a column to a layer of the geofile.

    Args:
        path (PathLike): Path to the geofile
        name (str): Name for the new column
        type (str): Column type.
        expression (str, optional): SQLite expression to use to update 
            the value. Defaults to None.
        layer (str, optional): The layer name. If None and the geofile
            has only one layer, that layer is used. Defaults to None.
        force_update (bool, optional): If the column already exists, execute 
            the update anyway. Defaults to False. 

    Raises:
        ex: [description]
    """

    ##### Init #####
    if type.upper() not in ['TEXT', 'NUMERIC', 'INTEGER', 'REAL', 'BLOB']:
        raise Exception(f"Type specified is not supported: {type}")
    path_p = Path(path)
    if layer is None:
        layer = get_only_layer(path_p)
    layerinfo_orig = get_layerinfo(path_p, layer)
    
    ##### Go! #####
    datasource = None
    try:
        #datasource = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_UPDATE)
        columns_upper = [column.upper() for column in layerinfo_orig.columns]
        if name.upper() not in columns_upper:
            # If column doesn't exist yet, create it
            #if name not in getlayerinfo(path_p, layer=layer).columns:
            sqlite_stmt = f'ALTER TABLE "{layer}" ADD COLUMN "{name}" {type}'
            ogr_util.vector_info(path=path_p, sql_stmt=sqlite_stmt, sql_dialect='SQLITE', readonly=False)
            #datasource.ExecuteSQL(sqlite_stmt, dialect='SQLITE')
        else:
            logger.warning(f"Column {name} existed already in {path_p}, layer {layer}")
            
        # If an expression was provided and update can be done, go for it...
        if(expression is not None 
           and (name not in layerinfo_orig.columns 
                or force_update is True)):
            sqlite_stmt = f'UPDATE "{layer}" SET "{name}" = {expression}'
            ogr_util.vector_info(path=path_p, sql_stmt=sqlite_stmt, sql_dialect='SQLITE', readonly=False)
            #datasource.ExecuteSQL(sqlite_stmt, dialect='SQLITE')
    finally:
        if datasource is not None:
            del datasource

def update_column(
        path: Union[str, 'os.PathLike[Any]'],
        name: str, 
        expression: str,
        layer: str = None,
        where: str = None):
    """
    Update a column from a layer of the geofile.

    Args:
        path (PathLike): Path to the geofile
        name (str): Name for the new column
        expression (str): SQLite expression to use to update 
            the value. 
        layer (str, optional): The layer name. If None and the geofile
            has only one layer, that layer is used. Defaults to None.
        layer (str, optional): SQL where clause to restrict the rows that will 
            be updated. Defaults to None.

    Raises:
        ex: [description]
    """

    ##### Init #####
    path_p = Path(path)
    if layer is None:
        layer = get_only_layer(path_p)
    layerinfo_orig = get_layerinfo(path_p, layer)
    
    ##### Go! #####
    datasource = None
    try:
        #datasource = gdal.OpenEx(str(path_p), nOpenFlags=gdal.OF_UPDATE)
        columns_upper = [column.upper() for column in layerinfo_orig.columns]
        if name.upper() not in columns_upper:
            # If column doesn't exist yet, error!
            raise Exception(f"Column {name} doesn't exist in {path_p}, layer {layer}")
            
        # If an expression was provided and update can be done, go for it...
        sqlite_stmt = f'UPDATE "{layer}" SET "{name}" = {expression}'
        if where is not None:
            sqlite_stmt += f"\n WHERE {where}"
        ogr_util.vector_info(path=path_p, sql_stmt=sqlite_stmt, sql_dialect='SQLITE', readonly=False)
        #datasource.ExecuteSQL(sqlite_stmt, dialect='SQLITE')
    finally:
        if datasource is not None:
            del datasource

def read_file(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        columns: List[str] = None,
        bbox = None,
        rows = None) -> gpd.GeoDataFrame:
    """
    Reads a file to a geopandas GeoDataframe. 
    
    The file format is detected based on the filepath extension.

    Args:
        path (file path): path to the file to read from
        layer (str, optional): The layer to read. Defaults to None,  
            then reads the only layer in the file or throws error.
        columns (List[str], optional): The (non-geometry) columns to read. 
            Defaults to None, then all columns are read.
        bbox ([type], optional): Read only geometries intersecting this bbox. 
            Defaults to None, then all rows are read.
        rows ([type], optional): Read only the rows specified. 
            Defaults to None, then all rows are read.
        ignore_geometry (bool, optional): True not to read/return the geomatry. 
            Defaults to False.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        gpd.GeoDataFrame: the data read.
    """
    result_gdf = _read_file_base(
            path=path,
            layer=layer,
            columns=columns,
            bbox=bbox,
            rows=rows,
            ignore_geometry=False)
    assert isinstance(result_gdf, gpd.GeoDataFrame)
    return result_gdf

def read_file_nogeom(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        columns: List[str] = None,
        bbox = None,
        rows = None) -> pd.DataFrame:
    """
    Reads a file to a pandas Dataframe. 
    
    The file format is detected based on the filepath extension.

    Args:
        path (file path): path to the file to read from
        layer (str, optional): The layer to read. Defaults to None,  
            then reads the only layer in the file or throws error.
        columns (List[str], optional): The (non-geometry) columns to read. 
            Defaults to None, then all columns are read.
        bbox ([type], optional): Read only geometries intersecting this bbox. 
            Defaults to None, then all rows are read.
        rows ([type], optional): Read only the rows specified. 
            Defaults to None, then all rows are read.
        ignore_geometry (bool, optional): True not to read/return the geomatry. 
            Defaults to False.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        pd.DataFrame: the data read.
    """
    result_gdf = _read_file_base(
            path=path,
            layer=layer,
            columns=columns,
            bbox=bbox,
            rows=rows,
            ignore_geometry=True)
    assert isinstance(result_gdf, pd.DataFrame)
    return result_gdf

def _read_file_base(
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        columns: List[str] = None,
        bbox = None,
        rows = None,
        ignore_geometry: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Reads a file to a pandas Dataframe. 
    
    The file format is detected based on the filepath extension.

    Args:
        path (file path): path to the file to read from
        layer (str, optional): The layer to read. Defaults to None,  
            then reads the only layer in the file or throws error.
        columns (List[str], optional): The (non-geometry) columns to read. 
            Defaults to None, then all columns are read.
        bbox ([type], optional): Read only geometries intersecting this bbox. 
            Defaults to None, then all rows are read.
        rows ([type], optional): Read only the rows specified. 
            Defaults to None, then all rows are read.
        ignore_geometry (bool, optional): True not to read/return the geomatry. 
            Defaults to False.

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: the data read.
    """
    # Init
    path_p = Path(path)
    ext_lower = path_p.suffix.lower()

    # For file multilayer types, if no layer name specified, check if there is only one layer in the file.
    if(ext_lower in ['.gpkg'] 
       and layer is None):
        listlayers = fiona.listlayers(str(path_p))
        if len(listlayers) == 1:
            layer = listlayers[0]
        else:
            raise Exception(f"File contains {len(listlayers)} layers: {listlayers}, but layer is not specified: {path}")

    # Depending on the extension... different implementations
    if ext_lower == '.shp':
        result_gdf = gpd.read_file(str(path_p), bbox=bbox, rows=rows, ignore_geometry=ignore_geometry)
    elif ext_lower == '.geojson':
        result_gdf = gpd.read_file(str(path_p), bbox=bbox, rows=rows, ignore_geometry=ignore_geometry)
    elif ext_lower == '.gpkg':
        result_gdf = gpd.read_file(str(path_p), layer=layer, bbox=bbox, rows=rows, ignore_geometry=ignore_geometry)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

    # If columns to read are specified... filter non-geometry columns 
    # case-insensitive 
    if columns is not None:
        columns_upper = [column.upper() for column in columns]
        columns_upper.append('GEOMETRY')
        columns_to_keep = [col for col in result_gdf.columns if (col.upper() in columns_upper)]
        result_gdf = result_gdf[columns_to_keep]
    
    # assert to evade pyLance warning 
    assert isinstance(result_gdf, pd.DataFrame) or isinstance(result_gdf, gpd.GeoDataFrame) 
    return result_gdf

def read_file_sql(
        path: Union[str, 'os.PathLike[Any]'],
        sql_stmt: str,
        sql_dialect: str = 'SQLITE',
        layer: str = None,
        ignore_geometry: bool = False) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Reads a file to a GeoPandas GeoDataFrame, using an sql statement to filter 
    the data. 

    Args:
        path (file path): path to the file to read from
        sql_stmt (str): sql statement to use
        sql_dialect (str, optional): Sql dialect used. Defaults to 'SQLITE'.
        layer (str, optional): The layer to read. If no layer is specified, 
            reads the only layer in the file or throws an Exception.
        ignore_geometry (bool, optional): True not to read/return the geomatry. 
            Defaults to False.

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: The data read.
    """

    # Check and init some parameters/variables
    path_p = Path(path)
    layer_list = None
    if layer is not None:
        layer_list = [layer]

    # Now we're ready to go!
    with tempfile.TemporaryDirectory() as tmpdir:
        # Execute sql againts file and write to temp file
        tmp_path = Path(tmpdir) / 'read_file_sql_tmp_file.gpkg'
        ogr_util.vector_translate(
                input_path=path_p,
                output_path=tmp_path,
                sql_stmt=sql_stmt,
                sql_dialect=sql_dialect,
                input_layers=layer_list,
                create_spatial_index=False)
            
        # Read and return result
        return _read_file_base(tmp_path, ignore_geometry=ignore_geometry)

def to_file(
        gdf: gpd.GeoDataFrame,
        path: Union[str, 'os.PathLike[Any]'],
        layer: str = None,
        force_multitype: bool = False,
        append: bool = False,
        append_timeout_s: int = 600,
        index: bool = True):
    """
    Writes a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to export to file.
        path (Union[str,): The file path to write to.
        layer (str, optional): The layer to read. If no layer is specified, 
            reads the only layer in the file or throws an Exception.
        force_multitype (bool, optional): force the geometry type to a multitype
            for file types that require one geometrytype per layer.
            Defaults to False.
        append (bool, optional): True to append to the file. Defaults to False.
        append_timeout_s (int, optional): The maximum timeout to wait when the 
            output file is already being written to by another process. 
            Defaults to 600.
        index (bool, optional): True to write the pandas index to the file as 
            well. Defaults to True.

    Raises:
        Exception: [description]
        Exception: [description]
    """
    # TODO: think about if possible/how to support adding optional parameter and pass them to next 
    # function, example encoding, float_format,...

    # Check input parameters
    path_p = Path(path)

    # If no layer name specified, use the filename (without extension)
    if layer is None:
        layer = Path(path_p).stem
    # If the dataframe is empty, log warning and return
    if len(gdf) <= 0:
        #logger.warn(f"Cannot write an empty dataframe to {filepath}.{layer}")
        return

    def write_to_file(
            gdf: gpd.GeoDataFrame,
            path: Path, 
            layer: str, 
            index: bool = True,
            force_multitype: bool = False,
            append: bool = False):

        # Change mode if append is true
        if append is True:
            if path.exists():
                mode = 'a'
            else:
                mode = 'w'
        else:
            mode = 'w'

        ext_lower = path.suffix.lower()
        if ext_lower == '.shp':
            if index is True:
                gdf_to_write = gdf.reset_index(drop=True)
            else:
                gdf_to_write = gdf
            gdf_to_write.to_file(str(path), mode=mode)
        elif ext_lower == '.gpkg':
            # Try to harmonize the geometrytype to one (multi)type, as GPKG
            # doesn't like > 1 type in a layer
            gdf_to_write = gdf.copy()
            gdf_to_write.geometry = geoseries_util.harmonize_geometrytypes(
                    gdf.geometry, force_multitype=force_multitype)
            gdf_to_write.to_file(str(path), layer=layer, driver="GPKG", mode=mode)
        else:
            raise Exception(f"Not implemented for extension {ext_lower}")

    # If no append, just write to output path
    if append is False:
        write_to_file(gdf=gdf, path=path_p, layer=layer, index=index, 
                force_multitype=force_multitype, append=append)
    else:
        # If append is asked, check if the fiona driver supports appending. If
        # not, write to temporary output file 
        
        # Remark: fiona pre-1.8.14 didn't support appending to geopackage. Once 
        # older versions becomes rare, dependency can be put to this version, and 
        # this code can be cleaned up...
        driver = get_driver(path_p)
        gdftemp_path = None
        gdftemp_lockpath = None
        if 'a' not in fiona.supported_drivers[driver]:
            # Get a unique temp file path. The file cannot be created yet, so 
            # only create a lock file to evade other processes using the same 
            # temp file name 
            gdftemp_path, gdftemp_lockpath = io_util.get_tempfile_locked(
                    base_filename='gdftemp',
                    suffix=path_p.suffix,
                    dirname='geofile_to_file')
            write_to_file(gdf, path=gdftemp_path, layer=layer, index=index, 
                    force_multitype=force_multitype)  
        
        # Files don't typically support having multiple processes writing 
        # simultanously to them, so use lock file to synchronize access.
        lockfile = Path(f"{str(path_p)}.lock")
        start_time = datetime.datetime.now()
        while(True):
            if io_util.create_file_atomic(lockfile) is True:
                try:
                    # If gdf wasn't written to temp file, use standard write-to-file
                    if gdftemp_path is None:
                        write_to_file(gdf=gdf, path=path_p, layer=layer, index=index, 
                                force_multitype=force_multitype, append=True)
                    else:
                        # If gdf was written to temp file, use append_to_nolock + cleanup
                        _append_to_nolock(src=gdftemp_path, dst=path_p, dst_layer=layer)
                        remove(gdftemp_path)
                        gdftemp_lockpath.unlink()
                finally:
                    lockfile.unlink()
                    return
            else:
                time_waiting = (datetime.datetime.now()-start_time).total_seconds()
                if time_waiting > append_timeout_s:
                    raise Exception(f"append_to_layer timeout of {append_timeout_s} reached, so stop trying!")
            
            # Sleep for a second before trying again
            time.sleep(1)
        
def get_crs(path: Union[str, 'os.PathLike[Any]']) -> pyproj.CRS:
    """
    Get the CRS (projection) of the file

    Args:
        path (PathLike): Path to the file.

    Returns:
        pyproj.CRS: The projection of the file
    """
    # TODO: seems like support for multiple layers in the file isn't here yet??? 
    with fiona.open(str(path), 'r') as geofile:
        return pyproj.CRS(geofile.crs)

def is_geofile(path: Union[str, 'os.PathLike[Any]']) -> bool:
    """
    Determines based on the filepath if this is a geofile.

    Args:
        path (PathLike): The file path.

    Returns:
        bool: True if it is a geo file.
    """
    return is_geofile_ext(Path(path).suffix)

def is_geofile_ext(file_ext: str) -> bool:
    """
    Determines based on the file extension if this is a geofile.

    Args:
        file_ext (str): the extension. 

    Returns:
        bool: True if it is a geofile.
    """
    file_ext_lower = file_ext.lower()
    if file_ext_lower in ('.shp', '.gpkg', '.geojson'):
        return True
    else:
        return False

def cmp(path1: Union[str, 'os.PathLike[Any]'], 
        path2: Union[str, 'os.PathLike[Any]']) -> bool:
    """
    Compare if two geofiles are identical. 

    For geofiles that use multiple files, all relevant files must be identical.
    Eg. for shapefiles, the .shp, .shx and .dbf file must be identical.

    Args:
        path1 (PathLike): path to the first file.
        path2 (PathLike): path to the second file.

    Returns:
        bool: True if the files are identical
    """
    # Check input parameters
    path1_p = Path(path1)
    path2_p = Path(path2)

    # For a shapefile, multiple files need to be compared
    if path1_p.suffix.lower() == '.shp':
        path2_noext, _ = os.path.splitext(path2_p)
        shapefile_base_suffixes = [".shp", ".dbf", ".shx"]
        path1_noext = path1_p.parent / path1_p.stem
        path2_noext = path2_p.parent / path2_p.stem
        for ext in shapefile_base_suffixes:
            if not filecmp.cmp(f"{str(path1_noext)}{ext}", f"{str(path2_noext)}{ext}"):
                logger.info(f"File {path1_noext}{ext} is different from {path2_noext}{ext}")
                return False
        return True
    else:
        return filecmp.cmp(str(path1_p), str(path2_p))
    
def copy(
        src: Union[str, 'os.PathLike[Any]'], 
        dst: Union[str, 'os.PathLike[Any]']):
    """
    Copies the geofile from src to dst. Is the source file is a geofile containing
    of multiple files (eg. .shp) all files are copied.

    Args:
        src (PathLike): the file to copy.
        dst (PathLike): the location to copy the file(s) to.
    """
    # Check input parameters
    src_p = Path(src)
    dst_p = Path(dst)

    # For a shapefile, multiple files need to be copied
    if src_p.suffix.lower() == '.shp':
        # If dest is a dir, just use move. Otherwise concat dest filepaths
        if dst_p.is_dir():
            for ext in shapefile_suffixes:
                srcfile = src_p.parent / f"{src_p.stem}{ext}"
                if srcfile.exists():
                    shutil.copy(str(srcfile), dst_p)
        else:
            for ext in shapefile_suffixes:
                srcfile = src_p.parent / f"{src_p.stem}{ext}"
                dstfile = dst_p.parent / f"{dst_p.stem}{ext}"
                if srcfile.exists():
                    shutil.copy(str(srcfile), dstfile)                
    else:
        return shutil.copy(str(src_p), dst_p)

def move(
        src: Union[str, 'os.PathLike[Any]'], 
        dst: Union[str, 'os.PathLike[Any]']):
    """
    Moves the geofile from src to dst. If the source file is a geofile containing
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
        # If dest is a dir, just use move. Otherwise concat dest filepaths
        if dst_p.is_dir():
            for ext in shapefile_suffixes:
                srcfile = src_p.parent / f"{src_p.stem}{ext}"
                if srcfile.exists():
                    shutil.move(str(srcfile), dst_p, copy_function=io_util.copyfile)
        else:
            for ext in shapefile_suffixes:
                srcfile = src_p.parent / f"{src_p.stem}{ext}"
                dstfile = dst_p.parent / f"{dst_p.stem}{ext}"
                if srcfile.exists():
                    shutil.move(str(srcfile), dstfile, copy_function=io_util.copyfile)                
    else:
        return shutil.move(str(src_p), dst_p, copy_function=io_util.copyfile)

def remove(path: Union[str, 'os.PathLike[Any]']):
    """
    Removes the geofile. Is it is a geofile composed of multiple files 
    (eg. .shp) all files are removed. 
    If .lock files are present, they are removed as well. 

    Args:
        path (PathLike): the file to remove
    """
    # Check input parameters
    path_p = Path(path)

    # If there is a lock file, remove it
    lockfile_path = path_p.parent / f"{path_p.name}.lock"
    lockfile_path.unlink(missing_ok=True)

    # For a shapefile, multiple files need to be removed
    if path_p.suffix.lower() == '.shp':
        for ext in shapefile_suffixes:
            curr_path = path_p.parent / f"{path_p.stem}{ext}"
            if curr_path.exists():
                curr_path.unlink()
    else:
        path_p.unlink()

def append_to(
        src: Union[str, 'os.PathLike[Any]'], 
        dst: Union[str, 'os.PathLike[Any]'],
        src_layer: str = None,
        dst_layer: str = None,
        force_output_geometrytype: Union[GeometryType, str] = None,
        create_spatial_index: bool = True,
        append_timeout_s: int = 600,
        verbose: bool = False):
    """
    Append src file to the dst file.

    remark: append is not supported for all filetypes in fiona/geopandas (0.8)
    so workaround via gdal needed.

    Args:
        src (Union[str,): source file path.
        dst (Union[str,): destination file path.
        src_layer (str, optional): source layer. Defaults to None.
        dst_layer (str, optional): destination layer. Defaults to None.
        force_output_geometrytype (GeometryType, optional): Geometry type. 
            to (try to) force the output to. Defaults to None.
        create_spatial_index (bool, optional): True to create a spatial index 
            on the destination file/layer. Defaults to True.
        append_timeout_s (int, optional): Timeout to use if the output file is
            being written to by another process already. Defaults to 600.
        verbose (bool, optional): True to write verbose output. Defaults to False.

    Raises:
        Exception: [description]
    """
    src_p = Path(src)
    dst_p = Path(dst)
    if force_output_geometrytype is not None:
        force_output_geometrytype = GeometryType(force_output_geometrytype)
    
    # Files don't typically support having multiple processes writing 
    # simultanously to them, so use lock file to synchronize access.
    lockfile = Path(f"{str(dst_p)}.lock")
    start_time = datetime.datetime.now()
    while(True):
        if io_util.create_file_atomic(lockfile) is True:
            try:
                # append
                _append_to_nolock(
                        src=src_p, 
                        dst=dst_p,
                        src_layer=src_layer,
                        dst_layer=dst_layer,
                        force_output_geometrytype=force_output_geometrytype,
                        create_spatial_index=create_spatial_index,
                        verbose=verbose)
            finally:
                lockfile.unlink()
                return
        else:
            time_waiting = (datetime.datetime.now()-start_time).total_seconds()
            if time_waiting > append_timeout_s:
                raise Exception(f"append_to_layer timeout of {append_timeout_s} reached, so stop trying!")
        
        # Sleep for a second before trying again
        time.sleep(1)

def _append_to_nolock(
        src: Path, 
        dst: Path,
        src_layer: str = None,
        dst_layer: str = None,
        force_output_geometrytype: GeometryType = None,
        create_spatial_index: bool = True,
        verbose: bool = False):
    # append
    translate_info = ogr_util.VectorTranslateInfo(
            input_path=src,
            output_path=dst,
            translate_description=None,
            input_layers=src_layer,
            output_layer=dst_layer,
            transaction_size=200000,
            append=True,
            update=True,
            force_output_geometrytype=force_output_geometrytype,
            create_spatial_index=create_spatial_index,
            priority_class='NORMAL',
            force_py=True,
            verbose=verbose)
    ogr_util.vector_translate_by_info(info=translate_info)

def convert(
        src: Union[str, 'os.PathLike[Any]'], 
        dst: Union[str, 'os.PathLike[Any]'],
        src_layer: str = None,
        dst_layer: str = None,
        force_output_geometrytype: GeometryType = None):
    """
    Convert the source file to the destination file. File types will be 
    detected based on the file extensions.

    Args:
        src (PathLike): The source file path.
        dst (PathLike): The destination file path.
    """
    src_p = Path(src)
    dst_p = Path(dst)

    # If dest file exists already, remove it
    if dst_p.exists():
        remove(dst_p)

    # Convert
    _append_to_nolock(src_p, dst_p, src_layer, dst_layer, force_output_geometrytype)

def get_driver(path: Union[str, 'os.PathLike[Any]']) -> str:
    """
    Get the driver to use for the file extension of this filepath.

    Args:
        path (PathLike): The file path.

    Returns:
        str: The OGR driver name.
    """
    return get_driver_for_ext(Path(path).suffix)

def get_driver_for_ext(file_ext: str) -> str:
    """
    Get the driver to use for this file extension.

    Args:
        file_ext (str): The extentension.

    Raises:
        Exception: [description]

    Returns:
        str: The OGR driver name.
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

def to_multi_type(geometrytypename: str) -> str:
    """
    Map the input geometry type to the corresponding 'MULTI' geometry type...

    DEPRECATED, use to_multigeometrytype
    
    Args:
        geometrytypename (str): Input geometry type

    Raises:
        Exception: If input geometrytype is not known.

    Returns:
        str: Corresponding 'MULTI' geometry type
    """
    return geometry_util.GeometryType(geometrytypename).to_multitype.name

def to_generaltypeid(geometrytypename: str) -> int:
    """
    Map the input geometry type name to the corresponding geometry type id:
        * 1 = POINT-type
        * 2 = LINESTRING-type
        * 3 = POLYGON-type

    DEPRECATED, use to_primitivetypeid()

    Args:
        geometrytypename (str): Input geometry type

    Raises:
        Exception: If input geometrytype is not known.

    Returns:
        int: Corresponding geometry type id
    """
    return geometry_util.GeometryType(geometrytypename).to_primitivetype.value
