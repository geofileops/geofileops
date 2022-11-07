# -*- coding: utf-8 -*-
"""
Tests for functionalities in vector_util, regarding geometry operations.
"""

from pathlib import Path
import sys

import shapely.geometry as sh_geom

# Add path so the local geofileops packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops import geometry_util


def test_centerline_box():
    # Very simple rectangular box -> simple line as centerline
    rectangle_poly = sh_geom.box(0, 0, 10, 2)
    rectangle_centerline = geometry_util.centerline(rectangle_poly)
    assert rectangle_centerline is not None
    assert rectangle_centerline.wkt == "LINESTRING (0 1, 5 1, 10 1)"

    # Very simple square box -> cross as centerline
    rectangle_poly = sh_geom.box(0, 0, 10, 10)
    rectangle_centerline = geometry_util.centerline(rectangle_poly)
    assert rectangle_centerline is not None
    assert rectangle_centerline.wkt == (
        "MULTILINESTRING ((5 5, 10 5), (5 5, 5 10), (5 0, 5 5), (0 5, 5 5))"
    )


def test_centerline_poly():

    # More complicated rectangle
    rectangle_poly = sh_geom.Polygon(shell=[(0, 0), (10, 0), (10, 2), (0, 2), (0, 0)])
    rectangle_centerline = geometry_util.centerline(rectangle_poly)
    assert rectangle_centerline is not None
    assert rectangle_centerline.wkt == "LINESTRING (0 1, 5 1, 10 1)"
