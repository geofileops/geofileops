# import datetime
import logging
from typing import Optional

import shapely
import pygeoops

# from pygeoops import _difference as _difference
# from pygeoops import _paramvalidation as paramvalidation

# Get a logger...
logger = logging.getLogger(__name__)


def gfo_difference_collection(
    geom_wkb: bytes,
    geom_to_subtract_wkb: bytes,
    keep_geom_type: int = 0,
    subdivide_coords: int = 1000,
) -> Optional[bytes]:
    """
    Applies the difference of geom_to_subtract on geom.

    If the input geometry has many points, they can be subdivided in smaller parts
    to potentially speed up processing as controlled by parameter `subdivide_coords`.
    This will result in extra collinear points being added to the boundaries of the
    output.

    Note that the geom_to_subtract_wkb won't be subdivided automatically, so if it
    can contain complex geometries as well you can use `gfo_subdivide` on it/them.

    Args:
        geom_wkb (bytes): geometry to substract geom_to_subtract_wkb from in wkb format.
        geom_to_subtract_wkb (bytes): geometry to substract from geom in wkb format.
            This can be a GeometryCollection containing many other geometries.
        keep_geom_type (int, optional): 1 to only retain geometries in the results of
            the same geometry type/dimension as the input. Eg. if input is a Polygon,
            remove LineStrings and Points from the difference result before returning.
            Defaults to 0.
        subdivide_coords (int, optional): if > 0, the input geometry will be
            subdivided to parts with about this number of points which can speed up
            processing for complex geometries. Subdividing can result in extra collinear
            points being added to the boundaries of the output. If <= 0, no subdividing
            is applied. Defaults to 1000.

    Returns:
        Optional[bytes]: return the difference. If geom was completely removed due to
            the difference applied, NULL is returned.
    """
    try:
        # Check/prepare input
        if geom_wkb is None:
            return None
        if geom_to_subtract_wkb is None:
            return geom_wkb

        # Extract wkb's, and return if empty
        geom = shapely.from_wkb(geom_wkb)
        if geom.is_empty:
            return geom_wkb
        geoms_to_subtract = shapely.from_wkb(geom_to_subtract_wkb)
        if geoms_to_subtract.is_empty:
            return geom_wkb
        del geom_wkb
        del geom_to_subtract_wkb

        # Check and convert booleanish int inputs to bool.
        keep_geom_type = _int2bool(keep_geom_type, "keep_geom_type")

    except Exception as ex:  # pragma: no cover
        # ex.with_traceback()
        logger.exception(f"Error in gfo_difference_collection: {ex}")
        raise

    try:
        # Apply difference
        result = pygeoops.difference_all_tiled(
            geom,
            geoms_to_subtract,
            keep_geom_type=keep_geom_type,
            subdivide_coords=subdivide_coords,
        )

        # If an empty result, return None
        # Remark: tried to return empty geometry an empty GeometryCollection, but
        # apparentle ST_IsEmpty of spatialite doesn't work (in combination with gpkg
        # and/or wkb?).
        if result is None or result.is_empty:
            return None

        return shapely.to_wkb(result)
    except Exception as ex:  # pragma: no cover
        # ex.with_traceback()
        logger.exception(f"Error in gfo_difference_collection: {ex}")
        return None


def gfo_reduceprecision(
    geom_wkb: bytes,
    gridsize: int,
    makevalid_first: int = 0,
) -> Optional[bytes]:
    """
    Reduces the precision of the geometry to the gridsize specified.

    By default, geometries use double precision coordinates (grid_size = 0). Coordinates
    will be rounded if a precision grid is less precise than the input geometry.
    Duplicated vertices will be dropped from lines and polygons for grid sizes greater
    than 0. Line and polygon geometries may collapse to empty geometries if all vertices
    are closer together than grid_size. Z values, if present, will not be modified.

    Note: unless parameter makevalid_first=1 is used, the input geometry should be
    geometrically valid. Unexpected results may occur if input geometry is not.

    Args:
        geom_wkb (bytes): geometry to reduce precision from in wkb format.
        gridsize (int): the size of the grid the coordinates of the ouput will be
            rounded to. Eg. 0.001 to keep 3 decimals. Value 0.0 doesn't change the
            precision.
        makevalid_first (int): if 1, the input is first made valid before reducing the
            precision.

    Returns:
        Optional[bytes]: return the geometry with the precision reduced.
    """
    try:
        # Check/prepare input
        if geom_wkb is None:
            return None

        # Extract wkb's, and return if empty
        geom = shapely.from_wkb(geom_wkb)
        if geom.is_empty:
            return geom_wkb
        del geom_wkb

    except Exception as ex:  # pragma: no cover
        # ex.with_traceback()
        logger.exception(f"Error in gfo_reduceprecision: {ex}")
        raise

    try:
        # If needed, apply makevalid first
        result = geom
        if makevalid_first:
            result = shapely.make_valid(result)

        # Apply set_precision
        result = shapely.set_precision(result, grid_size=gridsize)

        # If an empty result, return None
        # Remark: apparently ST_IsEmpty of spatialite doesn't work (in combination with
        # gpkg and/or wkb?).
        if result is None or result.is_empty:
            return None

        return shapely.to_wkb(result)

    except Exception as ex:  # pragma: no cover
        # ex.with_traceback()
        logger.exception(f"Error in gfo_reduceprecision: {ex}")
        return None


def gfo_subdivide(geom_wkb: bytes, coords: int = 1000):
    """
    Divide the input geometry to smaller parts using rectilinear lines.

    Args:
        geom_wkb (geometry): the geometry to subdivide in wkb format.
        coords (int): number of coordinates per subdivision to aim for. In the current
            implementation, coords will be the average number of coordinates the
            subdividions will consist of. If <= 0, no subdividing is applied.
            Defaults to 1000.

    Returns:
        geometry wkb: if geometry has < coords coordinates, the input geometry is
            returned. Otherwise the subdivisions as a GeometryCollection.
    """
    try:
        # Check/prepare input
        if geom_wkb is None:
            return None
        if coords <= 0:
            return geom_wkb

        # Extract wkb's, and return if empty
        geom = shapely.from_wkb(geom_wkb)
        if geom.is_empty:
            return geom_wkb
        del geom_wkb

    except Exception as ex:  # pragma: no cover
        # ex.with_traceback()
        logger.exception(f"Error in gfo_subdivide: {ex}")
        raise

    try:
        result = pygeoops.subdivide(geom, num_coords_max=coords)

        if result is None:
            return None
        if not hasattr(result, "__len__"):
            return shapely.to_wkb(result)
        if len(result) == 1:
            return shapely.to_wkb(result[0])

        return shapely.to_wkb(shapely.GeometryCollection(result.tolist()))

    except Exception as ex:  # pragma: no cover
        # ex.with_traceback()
        logger.exception(f"Error in gfo_subdivide: {ex}")
        return None


"""
class DifferenceAgg:
    def __init__(self):
        self.init_todo = True
        self.tmpdiff = None
        self.is_split = False
        self.geom_mbrp = None
        self.keep_geom_type_dimension = None
        self.num_coords_max = 1000

        # Some properties regarding progress writing to file
        self.enable_progress = False
        self.step_count = 0
        self.steps_last_progress = 0
        self.last_progress = datetime.datetime.now()

    def step(self, geom, geoms_to_subtract, keep_geom_type: int):
        try:
            # Init on first call
            if self.init_todo:
                self.init_todo = False
                if geom is None:
                    self.tmpdiff = None
                geom = shapely.from_wkb(geom)
                self.geom_mbrp = shapely.box(*geom.bounds)

                # Determine type/dimension to keep
                if keep_geom_type == 1:
                    keep_geom_type_bool = True
                elif keep_geom_type == 0:
                    keep_geom_type_bool = False
                else:
                    raise ValueError(
                        "Invalid value for keep_geom_type: only 0 (False) or 1 (True) "
                        "supported."
                    )
                self.keep_geom_type_dimension = (
                    paramvalidation.keep_geom_type2dimension(
                        keep_geom_type=keep_geom_type_bool, geometry=geom
                    )
                )

                # Split input geometry if needed
                self.tmpdiff = pygeoops.subdivide(geom, self.num_coords_max)

            # If the difference is already empty, no use to continue
            if self.tmpdiff is None:
                return

            # Apply difference
            geom_to_subtract = shapely.from_wkb(geoms_to_subtract)
            self.tmpdiff = _difference._difference_intersecting(
                self.tmpdiff,
                geom_to_subtract,
                keep_geom_type=self.keep_geom_type_dimension,
            )

            # Check for empty results
            self.tmpdiff = self.tmpdiff[~shapely.is_empty(self.tmpdiff)]
            if len(self.tmpdiff) == 0:
                self.tmpdiff = None

            # Write some progress debugging if enabled
            if self.enable_progress:
                self.write_progress()

        except Exception as ex:
            # ex.with_traceback()
            print(ex)

    def finalize(self):
        try:
            if (
                self.tmpdiff is None
                or len(self.tmpdiff) == 0
                or shapely.is_empty(self.tmpdiff).all()
            ):
                return None
            elif len(self.tmpdiff) == 1:
                return shapely.to_wkb(self.tmpdiff[0])
            else:
                return shapely.to_wkb(shapely.unary_union(self.tmpdiff))

        except Exception as ex:
            raise ex

    def write_progress(self):
        self.step_count += 1
        if self.step_count % 100 == 0:
            now = datetime.datetime.now()
            nb_steps_per_sec = (self.step_count - self.steps_last_progress) / (
                now - self.last_progress
            ).total_seconds()
            with open("c:/temp/progress.txt", "a") as file:
                file.write(f"self.progress: {self.step_count}, {nb_steps_per_sec}/s\n")
            self.steps_last_progress = self.step_count
            self.last_progress = now
"""


def _int2bool(value: int, variable_name: str) -> bool:
    if not isinstance(value, int):
        raise TypeError(
            f"{variable_name} must be int (0: False or 1: True), not {type(value)}"
        )
    if value not in [0, 1]:
        raise ValueError(f"{variable_name} has invalid value (0=False/1=True): {value}")

    if value == 0:
        return False
    else:
        return True
