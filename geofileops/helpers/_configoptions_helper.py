"""Helper class to access geofileops configuration options."""

import os
import tempfile
from pathlib import Path

from pyproj import CRS


class classproperty(property):
    def __get__(self, owner_self, owner_cls):  # noqa: ANN001, ANN204
        return self.fget(owner_cls)


class ConfigOptions:
    """Class to access the geofileops runtime configuration options.

    They are read from environement variables.
    """

    @classproperty
    def copy_layer_sqlite_direct(cls) -> bool:
        """Should copy_layer use sqlite directly when possible.

        This is significantly faster than using GDAL for large datasets. At the moment
        only used for .gpkg files.

        Returns:
            bool: True to use sqlite directly to copy layers. Defaults to True.
        """
        return get_bool("GFO_COPY_LAYER_SQLITE_DIRECT", default=True)

    @classproperty
    def io_engine(cls) -> str:
        """The IO engine to use."""
        io_engine = (
            os.environ.get("GFO_IO_ENGINE", default="pyogrio-arrow").strip().lower()
        )
        supported_values = ["pyogrio", "fiona", "pyogrio-arrow"]
        if io_engine not in supported_values:
            raise ValueError(
                f"invalid value for configoption <GFO_IO_ENGINE>: '{io_engine}', "
                f"should be one of {supported_values}"
            )

        return io_engine

    @classproperty
    def on_data_error(cls) -> str:
        """The preferred action when a data error occurs.

        Supported values (case insensitive):
            - "raise": raise an exception.
            - "warn": log a warning and continue.

        Note that the "warn" option is only very selectively supported: in many cases,
        an exception will still be raised.

        Returns:
            str: the preferred action when a data error occurs. Defaults to "raise".
        """
        value = os.environ.get("GFO_ON_DATA_ERROR")

        if value is None:
            return "raise"

        value_cleaned = value.strip().lower()
        supported_values = ["raise", "warn"]
        if value_cleaned not in supported_values:
            raise ValueError(
                f"invalid value for configoption <GFO_ON_DATA_ERROR>: '{value}', "
                f"should be one of {supported_values}"
            )

        return value_cleaned

    @classproperty
    def remove_temp_files(cls) -> bool:
        """Should temporary files be removed or not.

        Returns:
            bool: True to remove temp files. Defaults to True.
        """
        return get_bool("GFO_REMOVE_TEMP_FILES", default=True)

    @staticmethod
    def sliver_tolerance(crs: CRS | None) -> float:
        """Tolerance to use to filter out slivers from overlay operations.

        If 0.0, no sliver filtering is done. If negative, only slivers with tolerance
        abs(value) are retained in the output instead of filtering them out.

        If not set, the default depends on the ``crs``. If ``crs`` is a projected CRS,
        the default tolerance is 0.001 meters. If it is a geographic CRS, the default
        tolerance 1e-7 degrees. If ``crs`` is None or invalid, the default tolerance is
        0.0, so no sliver filtering is done.

        The slivers meant here are very small, often very narrow geometries that are
        created as a side-effect of overlay operations. Due to the limitations of
        finite-precision floating point arithmetic used in such operations, a point
        that is "snapped" on a line segment, is sometimes not exactly located on the
        line. When calculating e.g. an intersection, this can lead to very small sliver
        polygons being created.

        Most of the time, such slivers are not desired in the output. Hence, they are
        filtered out based on certain criteria by default.

        The basic algorythm used to determine if a geometry is a sliver is to use the
        GEOS `set_precision` algorythm with a small tolerance. If the polygon becomes
        NULL, because it is smaller/narrower than the tolerance, it is considered a
        sliver.

        Because the `set_precision` algorythm is relatively costly to apply, geometries
        are first pre-filtered with a less expensive filter: the average width:

        .. code_block::

            average_width(geom) = 2 * area(geom) / length(geom)

        This formula is an approximation that works well for long, narrow polygons, but
        it underestimates the width for square or round polygons. Some examples:

           - **a 10 x 10 meter square**: `2 * (10 * 10)  / (4 * 10) = 400 / 40 = 5`,
             which is an underestimation, as the real average width is 10.
           - **a 1 x 100 meter rectangle**:
             `2 * (1 * 100) / (2 * (1 + 100)) = 200 / 202 = ~0.99`, which is almost
             correct as the real average width is 1.

        The average width being underestimated for some shapes means that some
        geometries are marked as slivers even if they are not. Because the
        `average_width` check is only a pre-filter, this is not a problem: such
        geometries will still be retained if they pass the more precise `set_precision`
        check.

        Args:
            crs (CRS): The CRS of the geometries being processed. Used to determine
                the sliver tolerance. For projected CRSes, the tolerance is in the units
                of the CRS (e.g. meters). For geographic CRSes, the tolerance is in
                degrees.

        Returns:
            float: the sliver tolerance to be used. If 0.0, no sliver filtering is done.
        """
        try:
            tol_str = os.environ.get("GFO_SLIVER_TOLERANCE", None)
            if tol_str is not None:
                return float(tol_str)
            elif crs is None:
                return 0.0
            elif crs.is_projected:
                # Only found projected CRSs so far that use meters or feet, and for both
                # of those this tolerance is fine.
                return 0.001
            elif crs.is_geographic:
                return 1e-7
            else:
                return 0.0

        except Exception as ex:
            raise ValueError(
                "invalid value for configoption <GFO_SLIVER_TOLERANCE>: "
                "should be a number"
            ) from ex

    @classproperty
    def subdivide_check_parallel_fraction(cls) -> int:
        """For a file being checked in parallel, the fraction of features to check.

        Returns:
            int: The fraction of features to check for subdivision. Defaults to 5.
        """
        fraction = os.environ.get("GFO_SUBDIVIDE_CHECK_PARALLEL_FRACTION", default="5")

        return int(fraction)

    @classproperty
    def subdivide_check_parallel_rows(cls) -> int:
        """If a file has more rows, check if subdivide is needed in parallel.

        Returns:
            int: The minimum number of rows a file must have to check for subdivision in
                parallel. Defaults to 500000.
        """
        rows = os.environ.get("GFO_SUBDIVIDE_CHECK_PARALLEL_ROWS", default="500000")

        return int(rows)

    @classproperty
    def tmp_dir(cls) -> Path:
        """The temporary directory to use for processing.

        Returns:
            Path: The temporary directory path. Defaults to a system temp directory.
        """
        tmp_dir_str = os.environ.get("GFO_TMPDIR")
        if tmp_dir_str is None:
            tmpdir = Path(tempfile.gettempdir()) / "geofileops"
        elif tmp_dir_str.strip() == "":
            raise ValueError(
                "GFO_TMPDIR='' environment variable found which is not supported."
            )
        else:
            tmpdir = Path(tmp_dir_str.strip())

        tmpdir.mkdir(parents=True, exist_ok=True)
        return tmpdir

    @classproperty
    def worker_type(cls) -> str:
        """The type of workers to use for parallel processing.

        Supported values (case insensitive):
            - "threads": use threads when processing in parallel.
            - "processes": use processes when processing in parallel.
            - "auto": determine the type automatically.

        Returns:
            str: the type of workers to use. Defaults to "auto".
        """
        worker_type = os.environ.get("GFO_WORKER_TYPE", default="auto").strip().lower()
        supported_values = ["threads", "processes", "auto"]
        if worker_type not in supported_values:
            raise ValueError(
                f"invalid value for configoption <GFO_WORKER_TYPE>: '{worker_type}', "
                f"should be one of {supported_values}"
            )

        return worker_type


def get_bool(key: str, default: bool) -> bool:
    """Get the value for the environment variable ``key`` as a bool.

    Supported values (case insensitive):
       - True: "1", "YES", "TRUE"
       - False: "0", "NO", "FALSE"

    Args:
        key (str): the environement variable to read.
        default (bool): the value to return if the environement variable does not exist
            or if it is "".

    Raises:
        ValueError: if an invalid value is present in the environment variable.

    Returns:
        bool: True or False.
    """
    value = os.environ.get(key, default="")
    value_cleaned = value.strip().lower()

    # If the key is not defined, return default
    if value_cleaned == "":
        return default

    # Check the value
    if value_cleaned in ("1", "yes", "true"):
        return True
    elif value_cleaned in ("0", "no", "false"):
        return False
    else:
        raise ValueError(
            f"invalid value for bool configoption <{key}>: {value}, should be one of "
            "1, 0, YES, NO, TRUE, FALSE"
        )
