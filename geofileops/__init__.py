from pathlib import Path
import os

# Import geopandas here so the warning about pygeos <> shapely2 is given, but set
# USE_PYGEOS to avoid further warnings
import geopandas._compat as gpd_compat

if gpd_compat.USE_PYGEOS:
    os.environ["USE_PYGEOS"] = "1"
else:
    if gpd_compat.USE_SHAPELY_20:
        os.environ["USE_PYGEOS"] = "0"
    else:
        raise RuntimeError("geofileops needs either shapely2 or pygeos to be installed")

from geofileops.fileops import *  # noqa: F403, F401
from geofileops.geoops import *  # noqa: F403, F401
from geofileops.helpers.layerstyles import *  # noqa: F403, F401


def _get_version():
    version_path = Path(__file__).resolve().parent / "version.txt"
    with open(version_path, mode="r") as file:
        return file.readline()


__version__ = _get_version()
