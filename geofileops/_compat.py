"""Module with compatibility and dependency availability checks."""

import warnings

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from osgeo import gdal
from packaging import version

from geofileops.util import _ogr_util, _sqlite_util

gdal.UseExceptions()

# Do some pre-flight checks to ensure mandatory runtime dependencies are available.
sqlite3_spatialite_version_info = _sqlite_util.spatialite_version_info()
gdal_spatialite_version_info = _ogr_util.spatialite_version_info()

sqlite3_spatialite_version = sqlite3_spatialite_version_info["spatialite_version"]
gdal_spatialite_version = gdal_spatialite_version_info["spatialite_version"]
if (
    sqlite3_spatialite_version is None or sqlite3_spatialite_version == ""
):  # pragma: no cover
    warnings.warn("sqlite3 spatialite version could not be determined", stacklevel=1)
if gdal_spatialite_version is None or gdal_spatialite_version == "":  # pragma: no cover
    warnings.warn("gdal spatialite version could not be determined", stacklevel=1)
if sqlite3_spatialite_version != gdal_spatialite_version:  # pragma: no cover
    warnings.warn(
        "different spatialite versions loaded: "
        f"{sqlite3_spatialite_version=} vs {gdal_spatialite_version=}",
        stacklevel=1,
    )

# Determine the versions of the runtime dependencies
GDAL_GTE_38 = version.parse(gdal.__version__) >= version.parse("3.8")
GEOPANDAS_GTE_10 = version.parse(gpd.__version__) >= version.parse("1.0")
PANDAS_GTE_22 = version.parse(pd.__version__) >= version.parse("2.2")
PYOGRIO_GTE_07 = version.parse(pyogrio.__version__) >= version.parse("0.7")
SHAPELY_GTE_20 = version.parse(shapely.__version__) >= version.parse("2")
SPATIALITE_GTE_51 = version.parse(sqlite3_spatialite_version) >= version.parse("5.1")
