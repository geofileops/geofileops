"""Helper class to access the geofileops configuration options."""

import os
import tempfile
from pathlib import Path


class Options:
    """These are the configuration options supported by geofileops."""

    COPY_LAYER_SQLITE_DIRECT = "GFO_COPY_LAYER_SQLITE_DIRECT"
    """Enable option to copy data directly using sqlite in `copy_layer` when possible.

    This is significantly faster than having the data pass through GDAL for large
    datasets. It is only supported for .gpkg files and for very specific scenarios.
    Some limitations:
       - only `write_mode="append"`
       - explodecollections is not supported
       - no reprojection is supported
       - no field mapping is supported
       - ...

    Possible values:
        - **"TRUE"**: enable the option. This is the default.
        - **"FALSE"**: disable the option.
    """
    IO_ENGINE = "GFO_IO_ENGINE"
    """The IO engine to use.

    Possible values:
        - **"pyogrio"**: use the pyogrio library. This is the default.
        - **"fiona"**: use the fiona library.
    """
    ON_DATA_ERROR = "GFO_ON_DATA_ERROR"
    """The preferred action when a data error occurs.

    Possible values:
        - **"RAISE"**: raise an exception. This is the default.
        - **"WARN"**: log a warning and continue.
    """
    REMOVE_TEMP_FILES = "GFO_REMOVE_TEMP_FILES"
    """Should temporary files be removed or not.

    Possible values:
        - **"TRUE"**: remove temporary files. This is the default.
        - **"FALSE"**: do not remove temporary files.
    """
    SLIVER_TOLERANCE = "GFO_SLIVER_TOLERANCE"
    """Tolerance to use to filter out slivers from overlay operations.

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

    The value set should be a float representing the tolerance to use in the units of
    the spatial reference system  (SRS) used. If the tolerance set is 0.0, no sliver
    filtering is done. If not set, the tolerance defaults to 1e-8, or 10 nanometer for
    an SRS using meters as unit or up to 1 centimeter for an SRS using degrees as unit.
    """
    SUBDIVIDE_CHECK_PARALLEL_FRACTION = "GFO_SUBDIVIDE_CHECK_PARALLEL_FRACTION"
    """For a file being checked in parallel, the fraction of features to check.

    The value set should be an integer representing the fraction of features to check.
    If not set, defaults to 5, resulting in 20% of the features being checked.
    """
    SUBDIVIDE_CHECK_PARALLEL_ROWS = "GFO_SUBDIVIDE_CHECK_PARALLEL_ROWS"
    """For a file being checked in parallel, the number of rows to check.

    The value set should be an integer representing the minimum number of rows a file
    must have to check for subdivision in parallel. If not set, defaults to 500000.
    """
    TMP_DIR = "GFO_TMPDIR"
    """The temporary directory to use for temp files created during processing.

    The value set should be a valid directory path. If not set, a subdirectory
    "geofileops" created in the system temp directory is used.
    """
    WORKER_TYPE = "GFO_WORKER_TYPE"
    """The type of workers to use for parallel processing.

        Supported values (case insensitive):
            - "threads": use threads when processing in parallel.
            - "processes": use processes when processing in parallel.
            - "auto": determine the type automatically.

        Returns:
            str: the type of workers to use. Defaults to "auto".
    """


class classproperty(property):
    def __get__(self, owner_self, owner_cls):  # noqa: ANN001, ANN204
        return self.fget(owner_cls)


class ConfigOptions:
    """Class to access the geofileops runtime configuration options.

    They are read from environment variables.
    """

    @classproperty
    def copy_layer_sqlite_direct(cls) -> bool:
        """Should copy_layer use sqlite directly when possible.

        This is significantly faster than using GDAL for large datasets. It is only
        supported for .gpkg files and for specific scenarios like a straightforward
        "append".

        Returns:
            bool: True to use sqlite directly to copy layers. Defaults to True.
        """
        return get_bool(Options.COPY_LAYER_SQLITE_DIRECT, default=True)

    @classproperty
    def io_engine(cls) -> str:
        """The IO engine to use."""
        io_engine = os.environ.get(Options.IO_ENGINE, default="pyogrio").strip().lower()
        supported_values = ["pyogrio", "fiona"]
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
        value = os.environ.get(Options.ON_DATA_ERROR)

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
        return get_bool(Options.REMOVE_TEMP_FILES, default=True)

    @classproperty
    def subdivide_check_parallel_fraction(cls) -> int:
        """For a file being checked in parallel, the fraction of features to check.

        Returns:
            int: The fraction of features to check for subdivision. Defaults to 5.
        """
        fraction = os.environ.get(
            Options.SUBDIVIDE_CHECK_PARALLEL_FRACTION, default="5"
        )

        return int(fraction)

    @classproperty
    def subdivide_check_parallel_rows(cls) -> int:
        """If a file has more rows, check if subdivide is needed in parallel.

        Returns:
            int: The minimum number of rows a file must have to check for subdivision in
                parallel. Defaults to 500000.
        """
        rows = os.environ.get(Options.SUBDIVIDE_CHECK_PARALLEL_ROWS, default="500000")

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
        worker_type = (
            os.environ.get(Options.WORKER_TYPE, default="auto").strip().lower()
        )
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
