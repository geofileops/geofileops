# -*- coding: utf-8 -*-
"""
Module containing utilities regarding operations on geoseries.
"""

import logging
from typing import List

import geopandas as gpd
from geopandas.geoseries import GeoSeries
from shapely import geometry as sh_geom

from . import geometry_util
from .geometry_util import GeometryType, PrimitiveType, SimplifyAlgorithm

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# GeoDataFrame helpers
#-------------------------------------------------------------

def geometry_collection_extract(
        geoseries: gpd.GeoSeries,
        primitivetype: PrimitiveType) -> gpd.GeoSeries:
    '''
    # Apply the collection_extract
    return gpd.GeoSeries(
            [geometry_util.collection_extract(geom, primitivetype) for geom in geoseries])
    '''
    # Apply the collection_extract
    geoseries_copy = geoseries.copy()
    for index, geom in geoseries_copy.iteritems():
        geoseries_copy[index] = geometry_util.collection_extract(geom, primitivetype)
    assert isinstance(geoseries_copy, gpd.GeoSeries)
    return geoseries_copy

def get_geometrytypes(
        geoseries: gpd.GeoSeries,
        ignore_empty_geometries: bool = True) -> List[GeometryType]:
    """
    Determine the geometry types in the GeoDataFrame.

    In a GeoDataFrame, empty geometries are always treated as 
    geometrycollections. These are by default ignored. 

    Args:
        geoseries (gpd.GeoSeries): input geoseries.
        ignore_empty_geometries (bool, optional): True to ignore empty 
            geometries, as they are always stored as GeometryCollections by 
            GeoPandas. Defaults to True.

    Returns:
        List[GeometryType]: [description]
    """
    if ignore_empty_geometries is True:
        input_geoseries = geoseries[~geoseries.is_empty]
    else:
        input_geoseries = geoseries
    geom_types_2D = input_geoseries[~input_geoseries.has_z].geom_type.unique()
    geom_types_2D = [gtype for gtype in geom_types_2D if gtype is not None]
    geom_types_3D = input_geoseries[input_geoseries.has_z].geom_type.unique()
    geom_types_3D = ["3D " + gtype for gtype in geom_types_3D if gtype is not None]
    geom_types = geom_types_3D + geom_types_2D

    if len(geom_types) == 0:
        return [GeometryType.GEOMETRY]

    geometrytypes_list = [GeometryType[geom_type.upper()] for geom_type in geom_types]
    return geometrytypes_list

def harmonize_geometrytypes(
        geoseries: gpd.GeoSeries,
        force_multitype: bool = False) -> gpd.GeoSeries:
    """
    Tries to harmonize the geometries in the geoseries to one type.

    Eg. if Polygons and MultiPolygons are present in the geoseries, all 
    geometries are converted to MultiPolygons.

    Empty geometries are changed to None, because Empty geometries are always
    treated as GeometryCollections by GeoPandas.

    If they cannot be harmonized, the original series is returned...

    Args:
        geoseries (gpd.GeoSeries): The geoseries to harmonize.
        force_multitype (bool, optional): True to force all geometries to the 
            corresponding multitype. Defaults to False.

    Returns:
        gpd.GeoSeries: the harmonized geoseries if possible, otherwise the 
            original one.
    """
    # Get unique list of geometrytypes in gdf
    geometrytypes = get_geometrytypes(geoseries)

    # If already only one geometrytype...
    if len(geometrytypes) == 1: 
        if force_multitype is True:
            # If it is already a multitype, return
            if geometrytypes[0].is_multitype is True:
                return geoseries
            else:
                # Else convert to corresponding multitype
                return _harmonize_to_multitype(geoseries, geometrytypes[0].to_multitype)
        else:
            return geoseries
    elif(len(geometrytypes) == 2 
         and geometrytypes[0].to_primitivetype == geometrytypes[1].to_primitivetype):
        # There are two geometrytypes, but they are of the same primitive type, 
        # so can just be harmonized to the multitype
        return _harmonize_to_multitype(geoseries, geometrytypes[0].to_multitype)
    else:
        # Too difficult to harmonize, so just return 
        return geoseries

def _harmonize_to_multitype(
        geoseries: gpd.GeoSeries,
        dest_geometrytype: GeometryType) -> gpd.GeoSeries:
    # Loop over copy of the geoseries, and cast geometries that are not of the 
    # correct multitype yet... 
    # Remark: when not all indexers are filled in the geoseries, important that
    # the same indexers are used!!! 
    geoseries_copy = geoseries.copy()
    if dest_geometrytype is GeometryType.MULTIPOLYGON:
        for index, geom in geoseries_copy.iteritems():
            if geom is None or isinstance(geom, sh_geom.MultiPolygon):
                # If the geom is already ok, just continue
                continue
            if geom.is_empty:
                # If the geom is already ok, just continue
                geoseries_copy[index] = None
            elif isinstance(geom, sh_geom.Polygon):
                # If is is a Polygon, convert to MultiPolygon
                geoseries_copy[index] = sh_geom.MultiPolygon([geom])
            else:
                raise Exception(f"geom of geom_type {geom.geom_type} cannot be harmonized to {dest_geometrytype}")
    elif dest_geometrytype is GeometryType.MULTILINESTRING:
        for index, geom in geoseries_copy.iteritems():
            if geom is None or isinstance(geom, sh_geom.MultiLineString):
                # If the geom is already ok, just continue
                continue
            if geom.is_empty:
                # If the geom is already ok, just continue
                geoseries_copy[index] = None
            elif isinstance(geom, sh_geom.LineString):
                geoseries_copy[index] = sh_geom.MultiLineString([geom])
            else:
                raise Exception(f"geom of geom_type {geom.geom_type} cannot be harmonized to {dest_geometrytype}")
    elif dest_geometrytype is GeometryType.MULTIPOINT:
        for index, geom in geoseries_copy.iteritems():
            if geom is None or isinstance(geom, sh_geom.MultiPoint):
                # If the geom is already ok, just continue
                continue
            if geom.is_empty:
                # If the geom is already ok, just continue
                geoseries_copy[index] = None
            elif isinstance(geom, sh_geom.Point):
                geoseries_copy[index] = sh_geom.MultiPoint([geom])
            else:
                raise Exception(f"geom of geom_type {geom.geom_type} cannot be harmonized to {dest_geometrytype}")
    else:
        raise Exception(f"Unsupported destination GeometryType: {dest_geometrytype}")

    # assert to evade pyLance warning
    assert isinstance(geoseries_copy, gpd.GeoSeries)
    return geoseries_copy

def polygons_to_lines(geoseries: gpd.GeoSeries) -> gpd.GeoSeries:
    cardsheets_lines = []
    for cardsheet_poly in geoseries:
        cardsheet_boundary = cardsheet_poly.boundary
        if cardsheet_boundary.type == 'MultiLineString':
            for line in cardsheet_boundary:
                cardsheets_lines.append(line)
        else:
            cardsheets_lines.append(cardsheet_boundary)

    return gpd.GeoSeries(cardsheets_lines)    

def simplify_ext(
        geoseries: gpd.GeoSeries,
        algorithm: SimplifyAlgorithm,
        tolerance: float,
        lookahead: int = 8) -> gpd.GeoSeries:
    # If ramer-douglas-peucker, use standard geopandas algorithm
    if algorithm is SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER:
        return geoseries.simplify(tolerance=tolerance)
    else:
        # For other algorithms, use vector_util.simplify_ext()
        return gpd.GeoSeries(
                [geometry_util.simplify_ext(
                        geom, algorithm=algorithm, 
                        tolerance=tolerance, 
                        lookahead=lookahead) for geom in geoseries])
