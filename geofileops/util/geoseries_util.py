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
    
    # Apply the collection_extract
    return gpd.GeoSeries(
            [geometry_util.collection_extract(geom, primitivetype) for geom in geoseries])

def get_geometrytypes(geoseries: gpd.GeoSeries) -> List[GeometryType]:
    """
    Determine the geometry types in the GeoDataFrame.
    """
    geom_types_2D = geoseries[~geoseries.has_z].geom_type.unique()
    geom_types_2D = [gtype for gtype in geom_types_2D if gtype is not None]
    geom_types_3D = geoseries[geoseries.has_z].geom_type.unique()
    geom_types_3D = ["3D " + gtype for gtype in geom_types_3D if gtype is not None]
    geom_types = geom_types_3D + geom_types_2D

    if len(geom_types) == 0:
        return [GeometryType.GEOMETRY]

    geometrytypes_list = [GeometryType[geom_type.upper()] for geom_type in geom_types]
    return geometrytypes_list

def harmonize_geometrytypes(
        geoseries: gpd.GeoSeries,
        force_multitype: bool = False) -> gpd.GeoSeries:
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
    # Cast to destination geometrytype
    if dest_geometrytype is GeometryType.MULTIPOLYGON:
        return GeoSeries([sh_geom.MultiPolygon([geom]) 
                        if isinstance(geom, sh_geom.Polygon)  
                        else geom for geom in geoseries])
    elif dest_geometrytype is GeometryType.MULTIPOINT:
        return GeoSeries([sh_geom.MultiPoint([geom]) 
                        if isinstance(geom, sh_geom.Point)
                        else geom for geom in geoseries])
    elif dest_geometrytype is GeometryType.MULTILINESTRING:
        return GeoSeries([sh_geom.MultiLineString([geom]) 
                        if isinstance(geom, sh_geom.LineString) 
                        else geom for geom in geoseries])
    else:
        raise Exception(f"Unsupported geometrytype: {dest_geometrytype}")

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
