from pathlib import Path
import os
os.environ['USE_PYGEOS'] = '1'

from geofileops.fileops import *
from geofileops.geoops import *


def _get_version():
    version_path = Path(__file__).resolve().parent / "version.txt"
    with open(version_path, mode="r") as file:
        return file.readline()


__version__ = _get_version()
