# -*- coding: utf-8 -*-
"""
Tests for operations using GeoPandas.
"""

from pathlib import Path
import sys

import pytest

# Add path so the local geofileops packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geofileops as gfo
from geofileops import GeometryType
from geofileops.util import _geoops_ogr
from tests import test_helper
from tests.test_helper import DEFAULT_SUFFIXES


@pytest.mark.parametrize("suffix", DEFAULT_SUFFIXES)
def test_clip_by_geometry(tmp_path, suffix):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel", suffix=suffix)

    # Do operation
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    clip_wkt = (
        "Polygon ((156072 196691, 156036 196836, 156326 196927, 156368 196750, "
        "156072 196691))"
    )
    _geoops_ogr.clip_by_geometry(
        input_path=input_path, output_path=output_path, clip_geometry=clip_wkt
    )

    # Now check if the output file is correctly created
    assert output_path.exists()
    layerinfo_orig = gfo.get_layerinfo(input_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 22
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None


@pytest.mark.parametrize("suffix", DEFAULT_SUFFIXES)
def test_export_by_geometry(tmp_path, suffix):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-parcel", suffix=suffix)

    # Do operation
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"
    filter = (156036, 196691, 156368, 196927)
    _geoops_ogr.export_by_bounds(
        input_path=input_path, output_path=output_path, bounds=filter
    )

    # Now check if the output file is correctly created
    assert output_path.exists()
    layerinfo_orig = gfo.get_layerinfo(input_path)
    layerinfo_output = gfo.get_layerinfo(output_path)
    assert layerinfo_output.featurecount == 25
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)
    assert layerinfo_output.geometrytype == GeometryType.MULTIPOLYGON

    # Now check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None
