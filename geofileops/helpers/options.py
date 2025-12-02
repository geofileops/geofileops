"""Module to set geofileops configuration options."""

import os
from contextlib import AbstractContextManager
from typing import Literal


class _RestoreOriginalHandler(AbstractContextManager):
    def __init__(self, key: str, original_value: str | None) -> None:
        self.key = key
        self.original_value = original_value

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        if self.original_value is None:
            if self.key in os.environ:
                del os.environ[self.key]
        else:
            os.environ[self.key] = self.original_value


def copy_layer_sqlite_direct(enable: bool) -> _RestoreOriginalHandler:
    """Enable option to copy data directly using sqlite in `copy_layer` when possible.

    This option is enabled by default.

    This can be significantly faster than having the data pass through GDAL for large
    datasets.

    Some of the limitations:
       - only supported for Geopackage files
       - only `write_mode="append"`
       - explodecollections is not supported
       - no reprojection is supported
       - no field mapping is supported
       - ...

    Notes:
      - You can also set the option temporarily by using this function as a context
        manager.
      - You can also set the option by directly setting the environment variable
        `GFO_COPY_LAYER_SQLITE_DIRECT` to "TRUE" or "FALSE".

    Args:
        enable (bool): If True, this option is enabled.
    """
    key = "GFO_COPY_LAYER_SQLITE_DIRECT"
    original_value = os.environ.get(key)
    os.environ[key] = "TRUE" if enable else "FALSE"

    return _RestoreOriginalHandler(key, original_value)


def io_engine(
    engine: Literal["pyogrio-arrow", "pyogrio", "fiona"],
) -> _RestoreOriginalHandler:
    """Set the IO engine to use for reading and writing files.

    Possible options are:
        - **"pyogrio-arrow"** (default): use the pyogrio library via the arrow batch
          interface. The arrow will only be used if `pyarrow` is installed.
        - **"pyogrio"**: use the pyogrio library via the traditional interface.
        - **"fiona"**: use the fiona library. This is deprecated and support will be
          removed in a future release.

    Notes:
      - You can also set the option temporarily by using this function as a context
        manager.
      - You can also set the option by directly setting the environment variable
        `GFO_IO_ENGINE` to one of "PYOGRIO-ARROW", "PYOGRIO", or "FIONA".

    Args:
        engine (Literal["pyogrio-arrow", "pyogrio", "fiona"]): The IO engine to use.
    """
    key = "GFO_IO_ENGINE"
    original_value = os.environ.get(key)
    os.environ[key] = engine.upper()

    return _RestoreOriginalHandler(key, original_value)


def on_data_error(action: Literal["raise", "warn"]) -> _RestoreOriginalHandler:
    """Set the preferred action to take when a data error occurs.

    Possible options are:
        - **"raise"** (default): raise an exception.
        - **"warn"**: issue a warning and continue.

    Notes:
      - You can also set the option temporarily by using this function as a context
        manager.

    Args:
        action (Literal["raise", "warn"]): The action to take on data error.
    """
    key = "GFO_ON_DATA_ERROR"
    original_value = os.environ.get(key)
    os.environ[key] = action.upper()

    return _RestoreOriginalHandler(key, original_value)


def remove_temp_files(enable: bool) -> _RestoreOriginalHandler:
    """Enable or disable removal of temporary files created during operations.

    By default, temporary files are removed after the operation is complete.

    Notes:
      - You can also set the option temporarily by using this function as a context
        manager.
      - You can also set the option by directly setting the environment variable
        `GFO_REMOVE_TEMP_FILES` to "TRUE" or "FALSE".

    Args:
        enable (bool): If True, temporary files will be removed after operations.
    """
    key = "GFO_REMOVE_TEMP_FILES"
    original_value = os.environ.get(key)
    os.environ[key] = "TRUE" if enable else "FALSE"

    return _RestoreOriginalHandler(key, original_value)


def sliver_tolerance(tolerance: float) -> _RestoreOriginalHandler:
    """Tolerance to use to filter out slivers from overlay operations.

    The value set should be a float representing the tolerance to use in the units of
    the spatial reference system (SRS) used. If the tolerance set is 0.0, no sliver
    filtering is done. If not set, the tolerance defaults to 0.001 if the layers being
    processed are in a projected coordinate system, or 1e-7, if the data is in a
    geographic coordinate system.

    Slivers are typically very small, often very narrow geometries that are created
    as a side-effect of overlay operations. The cause of this are the limitations of
    finite-precision floating point arithmetic used typically in such operations. A
    point that is "snapped" on a line segment, is often not exactly on the line but e.g.
    a nanometer next to it. When calculating e.g. an intersection for this situation,
    this can lead to very narrow (~nanometer wide) sliver polygons being created.

    Most of the time, such slivers are not desired in the output. Hence, geofileops
    filters them out by default, based on certain criteria.

    The filter for the results to be retained in the output, so the geometries that
    are not slivers, is defined like this:
        WHERE average_width(geom) > {tolerance}
            OR set_precision(geom, {tolerance}) IS NOT NULL

    The average_width is calculated as:
        average_width(geom) = 2 * area(geom) / length(geom)

    This formula is an approximation that works well for square polygons (e.g. ).
    narrow slivers. However, for square or round geometries, this formula TODO.

    Notes:
      - You can also set the option temporarily by using this function as a context
        manager.
      - You can also set the option by directly setting the environment variable
        `GFO_SLIVER_TOLERANCE` to a string representing the tolerance value.

    Args:
        tolerance (float): The sliver tolerance value.
    """
    key = "GFO_SLIVER_TOLERANCE"
    original_value = os.environ.get(key)
    os.environ[key] = str(tolerance)

    return _RestoreOriginalHandler(key, original_value)
