# -*- coding: utf-8 -*-
"""
Module with helper functions to expand on some features of geopandas.
"""

import os
from pathlib import Path
from typing import List

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\anaconda3\envs\cropclassification3\Library\share\gdal"

import fiona
import geopandas as gpd

def read_file(filepath: str,
              layer: str = None,
              columns: List[str] = None,
              bbox = None,
              rows = None) -> gpd.GeoDataFrame:
    """
    Reads a file to a pandas dataframe. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support  adding optional parameter and pass them to next function, example encoding, float_format,...
    """
    # Init
    _, ext = os.path.splitext(filepath)
    ext_lower = ext.lower()

    # For file multilayer types, if no layer name specified, check if there is only one layer in the file.
    if(ext_lower in ['.gpkg'] 
       and layer is None):
        listlayers = fiona.listlayers(filepath)
        if len(listlayers) == 1:
            layer = listlayers[0]
        else:
            raise Exception(f"File contains {len(listlayers)} layers: {listlayers}, but layer is not specified: {filepath}")

    # Depending on the extension... different implementations
    if ext_lower == '.shp':
        return gpd.read_file(filepath, bbox=bbox, rows=rows)
    elif ext_lower == '.geojson':
        return gpd.read_file(filepath, bbox=bbox, rows=rows)
    elif ext_lower == '.gpkg':
        return gpd.read_file(filepath, layer=layer, bbox=bbox, rows=rows)
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")

def to_file(gdf: gpd.GeoDataFrame,
            filepath: str,
            layer: str = None,
            index: bool = True):
    """
    Reads a pandas dataframe to file. The fileformat is detected based on the filepath extension.

    # TODO: think about if possible/how to support adding optional parameter and pass them to next 
    # function, example encoding, float_format,...
    """
    # If no layer name specified, use the filename (without extension)
    if layer is None:
        layer = Path(filepath).stem
    # If the dataframe is empty, log warning and return
    if len(gdf) <= 0:
        #logger.warn(f"Cannot write an empty dataframe to {filepath}.{layer}")
        return

    _, ext = os.path.splitext(filepath)
    ext_lower = ext.lower()

    if ext_lower == '.shp':
        if index is True:
            gdf = gdf.reset_index(inplace=False)
        gdf.to_file(filepath)
    elif ext_lower == '.gpkg':
        gdf.to_file(filepath, layer=layer, driver="GPKG")
    else:
        raise Exception(f"Not implemented for extension {ext_lower}")
        
def get_crs(filepath):
    with fiona.open(filepath, 'r') as geofile:
        return geofile.crs

def is_geofile(filepath) -> bool:
    """
    Determines based on the filepath if this is a geofile.
    """
    _, file_ext = os.path.splitext(filepath)
    return is_geofile_ext(file_ext)

def is_geofile_ext(file_ext) -> bool:
    """
    Determines based on the file extension if this is a geofile.
    """
    file_ext_lower = file_ext.lower()
    if file_ext_lower in ('.shp', '.gpkg', '.geojson'):
        return True
    else:
        return False
