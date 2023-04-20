# -*- coding: utf-8 -*-
"""
Test if backwards compatibility for old API still works.
"""

from pathlib import Path
import sys

# Add path so the local geofileops packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geofileops
from geofileops import geofile
from tests import test_helper


def test_old_geofileops_api(tmp_path):
    input_path = test_helper.get_testfile("polygon-parcel")
    output_path = tmp_path / f"{input_path.stem}-output.gpkg"

    geofileops.buffer(input_path=input_path, output_path=output_path, distance=1)
    assert output_path.exists()

    input_gdf = geofile.read_file(path=input_path)
    assert len(input_gdf) > 0
