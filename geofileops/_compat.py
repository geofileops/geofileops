"""Module with compatibility and dependency availability checks."""

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from osgeo import gdal
from packaging import version

from geofileops.util import _sqlite_util

gdal.UseExceptions()

# Determine the versions of the runtime dependencies
# gdal.__version__ includes a "dev-..." suffix for master/development versions. This
# must be dropped for the version checks here.
GDAL_BASE_VERSION = version.parse(gdal.__version__.split("-")[0]).base_version
GDAL_GTE_38 = version.parse(GDAL_BASE_VERSION) >= version.parse("3.8")
GDAL_GTE_39 = version.parse(GDAL_BASE_VERSION) >= version.parse("3.9")
GDAL_GTE_3101 = version.parse(GDAL_BASE_VERSION) >= version.parse("3.10.1")
GDAL_ST_311 = version.parse(GDAL_BASE_VERSION) < version.parse("3.11")

GEOPANDAS_GTE_10 = version.parse(gpd.__version__) >= version.parse("1.0")
GEOPANDAS_110 = version.parse(gpd.__version__) == version.parse("1.1.0")
GEOS_GTE_313 = version.parse(shapely.geos_version_string) >= version.parse("3.13")
PANDAS_GTE_22 = version.parse(pd.__version__) >= version.parse("2.2")
PYOGRIO_GTE_07 = version.parse(pyogrio.__version__) >= version.parse("0.7")
PYOGRIO_GTE_08 = version.parse(pyogrio.__version__) >= version.parse("0.8")
SHAPELY_GTE_20 = version.parse(shapely.__version__) >= version.parse("2")

sqlite3_spatialite_version_info = _sqlite_util.spatialite_version_info()
sqlite3_spatialite_version = sqlite3_spatialite_version_info["spatialite_version"]
SPATIALITE_GTE_51 = version.parse(sqlite3_spatialite_version) >= version.parse("5.1")
