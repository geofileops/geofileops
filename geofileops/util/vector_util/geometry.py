# -*- coding: utf-8 -*-
"""
Module containing utilities regarding low level vector operations.
"""

import enum
import logging
import math
from typing import Any, List, Optional, Union

import geopandas as gpd
import numpy as np
import pygeos
import shapely.wkb as sh_wkb
import shapely.geometry as sh_geom
import shapely.ops as sh_ops
# Import simplification if it is available. Only throw exception in runtime...
try:
    import simplification.cutil as simplification
except ImportError as ex:
    None

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# Geometry helpers
#-------------------------------------------------------------

class GeometryType(enum.Enum):
    POINT = 'Point'
    LINESTRING = 'LineString'
    LINEARRING = 'LinearRing'
    POLYGON = 'Polygon'
    MULTIPOINT = 'MultiPoint'
    MULTILINESTRING = 'MultiLineString'
    MULTIPOLYGON = 'MultiPolygon'
    GEOMETRYCOLLECTION = 'GeometryCollection'

class PrimitiveType(enum.Enum):
    POINT = 'Point'
    LINESTRING = 'LineString'
    POLYGON = 'Polygon'

def to_geometrytype(geometrytype: Union[str, GeometryType]) -> GeometryType:
    """
    Cast a string to a GeometryTypes.

    Args:
        geometrytype (str): string to cast to GeometryTypes.

    Returns:
        GeometryTypes: the corresponding geometry type.
    """
    if isinstance(geometrytype, GeometryType):
        return geometrytype
    else:
        return GeometryType[geometrytype.upper()]

def to_primitivetype(geometrytype: Union[str, GeometryType]) -> PrimitiveType:
    """
    Get the primitive type for the geometry type specified.

    Args:
        geometrytype (Union[str, GeometryTypes]): the geometry type to convert.

    Raises:
        Exception: if a geometry type is passed that cannot be converted to a 
        primitive type.

    Returns:
        PrimitiveTypes: the corresponding primitive type.
    """
    # Prepare input parameters
    geometrytype_local = geometrytype
    if isinstance(geometrytype, str):
        geometrytype_local = to_geometrytype(geometrytype)

    # Lookup correct primitive type
    if geometrytype_local in [GeometryType.POINT, GeometryType.MULTIPOINT]:
        return PrimitiveType.POINT
    elif geometrytype_local in [GeometryType.LINESTRING, GeometryType.MULTILINESTRING]:
        return PrimitiveType.LINESTRING
    elif geometrytype_local in [GeometryType.POLYGON, GeometryType.MULTIPOLYGON]:
        return PrimitiveType.POLYGON
    else:
        raise Exception(f"No primitive type available for geometrytype {geometrytype_local}")

def to_primitivetypeid(primitivetype: PrimitiveType) -> int:
    if primitivetype == PrimitiveType.POINT:
        return 1
    elif primitivetype == PrimitiveType.LINESTRING:
        return 2
    elif primitivetype == PrimitiveType.POLYGON:
        return 3
    else:
        raise Exception(f"Not an existing primitivetype: {primitivetype}")

def collection_extract(
        geom: sh_geom.base.BaseGeometry,
        primitivetype: PrimitiveType) -> Optional[sh_geom.base.BaseGeometry]:
    """
    Extract only the features of the specified primitive type from the geometry.

    Args:
        geom (sh_geom.base.BaseGeometry): geometry to extract from.
        primitivetype (PrimitiveTypes): primitive types to extract.

    Returns:
        Optional[sh_geom.base.BaseGeometry]: a geometry containing only the 
            features from geom of the primitive type specified. 
    """

    # Extract the polygons from the multipolygon, but store them as multipolygons anyway
    returngeoms = collection_extract_to_list(geom, primitivetype=primitivetype)
    if len(returngeoms) > 0:
        return collect(returngeoms)
    else:
        return None

def collection_extract_to_list(
        geom: sh_geom.base.BaseGeometry,
        primitivetype: PrimitiveType) -> List[sh_geom.base.BaseGeometry]:
    """
    Extracts the geometries from the input geom that comply with the 
    primitive_type specified and returns them as a list.

    Args:
        geom (sh_geom.base.BaseGeometry): geometry to extract the polygons 
            from.
        primitivetype (GeometryPrimitiveTypes): the primitive type to extract
            from the input geom.

    Raises:
        Exception: if in_geom is an unsupported geometry type.  

    Returns:
        List[sh_geom.base.BaseGeometry]: List of primitive geometries, only 
            containing the primitive type specified.
    """
    # Extract the polygons from the multipolygon, but store them as multipolygons anyway
    if isinstance(geom, sh_geom.Point):
        if primitivetype == PrimitiveType.POINT:
            return [geom]
    elif isinstance(geom, sh_geom.MultiPoint):
        if primitivetype == PrimitiveType.POINT:
            return list(geom)
    elif isinstance(geom, sh_geom.LineString):
        if primitivetype == PrimitiveType.LINESTRING:
            return [geom]
    elif isinstance(geom, sh_geom.MultiLineString):
        if primitivetype == PrimitiveType.LINESTRING:
            return list(geom)
    elif isinstance(geom.geom_type, sh_geom.Polygon):
        if primitivetype == PrimitiveType.POLYGON:
            return [geom]
    elif isinstance(geom, sh_geom.MultiPolygon):
        if primitivetype == PrimitiveType.POLYGON:
            return list(geom)
    elif isinstance(geom.geom_type, sh_geom.GeometryCollection):
        returngeoms = []
        for geom in sh_geom.GeometryCollection(geom):
            returngeoms.append(collection_extract(geom, primitivetype=primitivetype))
        if len(returngeoms) > 0:
            return returngeoms

    return []

def collect(
        geometry_list_cleaned: List[sh_geom.base.BaseGeometry]) -> Optional[sh_geom.base.BaseGeometry]:
    """
    Collect a list of geometries to one geometry. 
    
    Examples:
      * if the list contains only Polygon's, returns a MultiPolygon. 
      * if the list contains different types, returns a GeometryCollection.

    Args:
        geometry_list (List[sh_geom.base.BaseGeometry]): [description]

    Raises:
        Exception: raises an exception if one of the input geometries is of an 
            unknown type. 

    Returns:
        Optional[sh_geom.base.BaseGeometry]: the result
    """
    # First remove all None geometries in the input list
    geometry_list_cleaned = [geometry for geometry in geometry_list_cleaned if geometry is not None]

    # If the list is empty or contains only 1 element, it is easy...
    if geometry_list_cleaned is None or len(geometry_list_cleaned) == 0: 
        return None
    elif len(geometry_list_cleaned) == 1:
        return geometry_list_cleaned[0]
    
    # Loop over all elements in the list, and determine the appropriate geometry type to create
    result_collection_type = GeometryType[geometry_list_cleaned[0].geom_type.upper()]
    for geom in geometry_list_cleaned:
        # If it is the same as the collection_geom_type, continue checking
        if geom.geom_type.upper() == result_collection_type.name:
            continue
        else:
            # If multiple types in the list, result becomes a geometrycollection
            result_collection_type = GeometryType.GEOMETRYCOLLECTION
            break
    
    # Now we can create the collection
    if result_collection_type == GeometryType.POINT:
        return sh_geom.MultiPoint(geometry_list_cleaned)
    elif result_collection_type == GeometryType.LINESTRING:
        return sh_geom.MultiLineString(geometry_list_cleaned)
    elif result_collection_type == GeometryType.POLYGON:
        return sh_geom.MultiPolygon(geometry_list_cleaned)
    elif result_collection_type == GeometryType.GEOMETRYCOLLECTION:
        return sh_geom.GeometryCollection(geometry_list_cleaned)
    else:
        raise Exception(f"Unsupported geometry type: {result_collection_type}")

def make_valid(geometry: sh_geom.base.BaseGeometry) -> sh_geom.base.BaseGeometry:
    """
    Make a geometry valid.

    Args:
        geometry (Optional[sh_geom.base.BaseGeometry]): A (possibly) invalid geometry.

    Returns:
        Optional[sh_geom.base.BaseGeometry]: The fixed geometry.
    """
    return sh_wkb.loads(pygeos.io.to_wkb(pygeos.make_valid(pygeos.io.from_shapely(geometry))))

def numberpoints(geometry: sh_geom.base.BaseGeometry) -> int:
    """
    Calculates the total number of points in a geometry.

    Args:
        geometry (sh_geom.base.BaseGeometry): the geometry to count the point of.

    Returns:
        int: the number of points in the geometry.
    """
    # If it is a multi-part, recursively call numberpoints for all parts. 
    if isinstance(geometry, sh_geom.base.BaseMultipartGeometry):
        nb_points = 0
        for geom in geometry:
            nb_points += numberpoints(geom)
        return nb_points
    elif isinstance(geometry, sh_geom.Polygon):
        # If it is a polygon, calculate number for exterior and interior rings. 
        nb_points = len(geometry.exterior.coords)
        for ring in geometry.interiors:
            nb_points += len(ring.coords)
        return nb_points
    else:
        # For other types, it is just the number of coordinates.
        return len(geometry.coords)

def remove_inner_rings(
        geometry,
        min_area_to_keep: float = None):
    
    # First check if the geom is None...
    if geometry is None:
        return None
    if geometry.type not in ('Polygon', 'Multipolgon'):
        raise Exception(f"remove_inner_rings is not possible with geometry.type: {geometry.type}, geometry: {geometry}")

    #if geometry.area > 91000 and geometry.area < 92000:
    #    logger.info("test")

    # If all inner rings need to be removed...
    if min_area_to_keep is None or min_area_to_keep == 0.0:
        # If there are no interior rings anyway, just return input
        if len(geometry.interiors) == 0:
            return geometry
        else:
            # Else create new polygon with only the exterior ring
            return sh_ops.Polygon(geometry.exterior)
    
    # If only small rings need to be removed... loop over them
    ring_coords_to_keep = []
    small_ring_found = False
    for ring in geometry.interiors:
        if abs(sh_ops.Polygon(ring).area) <= min_area_to_keep:
            small_ring_found = True
        else:
            ring_coords_to_keep.append(ring.coords)
    
    # If no small rings were found, just return input
    if small_ring_found == False:
        return geometry
    else:
        return sh_ops.Polygon(geometry.exterior.coords, ring_coords_to_keep)

class SimplifyAlgorithm(enum.Enum):
    RAMER_DOUGLAS_PEUCKER = 'rdp'
    LANG = 'lang'
    VISVALINGAM_WHYATT = 'vw'

def simplify_ext(
        geometry: sh_geom.base.BaseGeometry,
        tolerance: float,
        algorithm: SimplifyAlgorithm = SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
        lookahead: int = 8,
        preserve_topology: bool = False,
        keep_points_on: sh_geom.base.BaseGeometry = None) -> Optional[sh_geom.base.BaseGeometry]:
    """
    Simplify the geometry, with extended options.

    Args:
        geometry (shapely geometry): the geometry to simplify
        tolerance (float): mandatory for the following algorithms:  
            * RAMER_DOUGLAS_PEUCKER: distance to use as tolerance  
            * LANG: distance to use as tolerance
            * VISVALINGAM_WHYATT: area to use as tolerance
        algorithm (SimplifyAlgorithm, optional): algorithm to use.
            Defaults to SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER.
        lookahead (int, optional): the number of points to consider for removing
            in a moving window. Used for LANG algorithm. Defaults to 8.
        preserve_topology (bool, optional): True to (try to) return valid 
            geometries as result. Defaults to False.
        keep_points_on (BaseGeometry], optional): point of the geometry to 
            that intersect with these geometries are not removed. Defaults to None.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]

    Returns:
        sh_geom.base.BaseGeometry: The simplified version of the geometry.
    """
    # Init
    if algorithm in [SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
                     SimplifyAlgorithm.VISVALINGAM_WHYATT]:
        try:
            import simplification.cutil as simplification
        except ImportError as ex:
            raise ImportError(f"To use simplify_ext using rdp or vw, first do: 'pip install simplification'") from ex

    # Define some inline funtions 
    # Apply the simplification
    def simplify_polygon(polygon: sh_geom.Polygon) -> sh_geom.Polygon:
        # Simplify all rings
        exterior_simplified = simplify_coords(polygon.exterior.coords)
        interiors_simplified = []
        for interior in polygon.interiors:
            interior_simplified = simplify_coords(interior.coords)
            
            # If simplified version is ring, add it
            if(interior_simplified is not None 
               and len(interior_simplified) >= 3):
                interiors_simplified.append(interior_simplified) 
            elif preserve_topology:
                # If topology needs to be preserved, keep original ring
                interiors_simplified.append(interior.coords)              

        return sh_geom.Polygon(exterior_simplified, interiors_simplified)

    def simplify_linestring(linestring: sh_geom.LineString) -> sh_geom.LineString:
        # If the linestring cannot be simplified, return it
        if linestring is None or len(linestring.coords) <= 2:
            return linestring
        
        # Simplify
        coords_simplified = simplify_coords(linestring.coords)

        # If preserve_topology is True and the result is no line anymore, return original line
        if(preserve_topology is True
           and (coords_simplified is None or len(coords_simplified) < 2)):
            return linestring
        else:
            return sh_geom.LineString(coords_simplified)

    def simplify_coords(coords: list) -> List[Any]:
        # Determine the indexes of the coordinates to keep after simplification
        if algorithm is SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER:
            coords_simplify_idx = simplification.simplify_coords_idx(coords, tolerance)
        elif algorithm is SimplifyAlgorithm.VISVALINGAM_WHYATT:
            coords_simplify_idx = simplification.simplify_coords_vw_idx(coords, tolerance)
        elif algorithm is SimplifyAlgorithm.LANG:
            coords_simplify_idx = simplify_coords_lang_idx(coords, tolerance, lookahead=lookahead)    
        else:
            raise Exception(f"Unsupported algorithm: {algorithm}, supported: {SimplifyAlgorithm}")

        coords_on_border_idx = []
        if keep_points_on is not None:
            coords_gdf = gpd.GeoDataFrame(geometry=list(sh_geom.MultiPoint(coords)))
            coords_on_border_series = coords_gdf.intersects(keep_points_on)
            coords_on_border_idx = np.array(coords_on_border_series.index[coords_on_border_series]).tolist()

        # Extracts coordinates that need to be kept
        coords_to_keep = sorted(set(coords_simplify_idx + coords_on_border_idx))
        coords_simplified = np.array(coords)[coords_to_keep].tolist()
        return coords_simplified
    
    # Loop over the rings, and simplify them one by one...
    # If the geometry is None, just return...
    if geometry is None:
        return None
    elif isinstance(geometry, sh_geom.Point):
        # Point cannot be simplified
        return geometry
    elif isinstance(geometry, sh_geom.MultiPoint):
        # MultiPoint cannot be simplified
        return geometry
    elif isinstance(geometry, sh_geom.LineString):
        result_geom = simplify_linestring(geometry)
    elif isinstance(geometry, sh_geom.Polygon):
        result_geom = simplify_polygon(geometry)
    elif isinstance(geometry, sh_geom.base.BaseMultipartGeometry):
        # If it is a multi-part, recursively call simplify for all parts. 
        simplified_geometries = []
        for geom in geometry:
            simplified_geometries.append(simplify_ext(geom, 
                    tolerance=tolerance, 
                    algorithm=algorithm, lookahead=lookahead, 
                    preserve_topology=preserve_topology, 
                    keep_points_on=keep_points_on))
        result_geom = collect(simplified_geometries)
    else:
        raise Exception(f"Unsupported geom_type: {geometry.geom_type}, {geometry}")

    return make_valid(result_geom)

def simplify_coords_lang(
        coords: Union[np.ndarray, list],
        tolerance: float,
        lookahead: int) -> Union[np.ndarray, list]:
    """
    Simplify a line using lang algorithm.

    Args:
        coords (Union[np.ndarray, list]): list of coordinates to be simplified.
        tolerance (float): distance tolerance to use.
        lookahead (int, optional): the number of points to consider for removing
            in a moving window. Defaults to 8.

    Returns:
        Return the coordinates kept after simplification. 
        If input coords is np.ndarray, returns np.ndarray, otherwise returns a list.
    """

    # Init variables 
    if isinstance(coords, np.ndarray):
        coords_arr = coords
    else:
        coords_arr = np.array(list(coords))

    # Determine the coordinates that need to be kept 
    coords_to_keep_idx = simplify_coords_lang(
            coords=coords_arr,
            tolerance=tolerance,
            lookahead=lookahead)
    coords_simplified_arr = coords_arr[coords_to_keep_idx]

    # If input was np.ndarray, return np.ndarray, otherwise list
    if isinstance(coords, np.ndarray):
        return coords_simplified_arr
    else:
        return coords_simplified_arr.tolist()

def simplify_coords_lang_idx(
        coords: Union[np.ndarray, list],
        tolerance: float,
        lookahead: int = 8) -> Union[np.ndarray, list]:
    """
    Simplify a line using lang algorithm and return the coordinate indexes to 
    be kept.

    Inspiration for the implementation came from:
        * https://github.com/giscan/Generalizer/blob/master/simplify.py
        * https://github.com/keszegrobert/polyline-simplification/blob/master/6.%20Lang.ipynb
        * https://web.archive.org/web/20171005193700/http://web.cs.sunyit.edu/~poissad/projects/Curve/about_algorithms/lang.php

    Args:
        coords (Union[np.ndarray, list]): list of coordinates to be simplified.
        tolerance (float): distance tolerance to use.
        lookahead (int, optional): the number of points to consider for removing
            in a moving window. Defaults to 8.

    Returns:
        Return the indexes of coordinates that need to be kept after 
        simplification. 
        If input coords is np.ndarray, returns np.ndarray, otherwise returns a list.  
    """
    
    def point_line_distance(point_x, point_y, line_x1, line_y1, line_x2, line_y2):
        return abs((line_x2-line_x1)*(line_y1-point_y)-(line_x1-point_x)*(line_y2-line_y1)) / math.sqrt((line_x2-line_x1)*(line_x2-line_x1)+(line_y2-line_y1)*(line_y2-line_y1))

    # Init variables 
    if isinstance(coords, np.ndarray):
        line_arr = coords
    else:
        line_arr = np.array(list(coords))

    # Prepare lookahead
    nb_points = len(line_arr)
    if lookahead == -1:
        window_size = nb_points-1
    else:
        window_size = min(lookahead, nb_points-1)

    #mask = np.ones(nb_points), dtype='bool')
    mask = np.zeros(nb_points, dtype='bool')
    mask[0] = True
    window_start = 0
    window_end = window_size
    
    # Apply simplification till the window_start arrives at the last point.
    ready = False
    while ready is False:
        # Check if all points between window_start and window_end are within  
        # tolerance distance to the line (window_start, window_end).
        all_points_in_tolerance = True
        for i in range(window_start+1, window_end):
            distance = point_line_distance(
                    line_arr[i, 0], line_arr[i, 1],
                    line_arr[window_start, 0], line_arr[window_start, 1],
                    line_arr[window_end, 0], line_arr[window_end, 1])
            # If distance is nan (= linepoint1 == linepoint2) or > tolerance
            if math.isnan(distance) or distance > tolerance:
                all_points_in_tolerance = False
                break

        # If not all the points are within the tolerance distance... 
        if not all_points_in_tolerance:
            # Move window_end to previous point, and try again
            window_end -= 1
        else:
            # All points are within the tolerance, so they can be masked
            mask[window_end] = True
            #mask[window_start+1:window_end-1] = False

            # Move the window forward
            window_start = window_end
            if window_start == nb_points - 1:
                ready = True
            window_end = window_start + window_size
            if window_end >= nb_points: 
                window_end = nb_points - 1
    
    # Prepare result: convert the mask to a list of indices of points to keep. 
    idx_to_keep_arr = mask.nonzero()[0]

    # If input was np.ndarray, return np.ndarray, otherwise list
    if isinstance(coords, np.ndarray):
        return idx_to_keep_arr
    else:
        return idx_to_keep_arr.tolist()
