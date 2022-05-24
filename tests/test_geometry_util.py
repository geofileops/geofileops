# -*- coding: utf-8 -*-
"""
Tests for functionalities in vector_util, regarding geometry operations.
"""

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pygeos
import pytest
import shapely.geometry as sh_geom

# Add path so the local geofileops packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import geofileops as gfo
from geofileops import geometry_util
from geofileops import GeometryType, PrimitiveType
from geofileops.util import grid_util
from tests import test_helper


def test_geometrytype():
    # Creating a GeometryType from None is invalid
    with pytest.raises(ValueError, match="None is not a valid GeometryType"):
        geometrytype = GeometryType(None)

    # Create different types of Geometrytype
    geometrytype = GeometryType(3)
    assert geometrytype is GeometryType.POLYGON
    geometrytype = GeometryType("PoLyGoN")
    assert geometrytype is GeometryType.POLYGON
    geometrytype = GeometryType(GeometryType.POLYGON)
    assert geometrytype is GeometryType.POLYGON

    # Test to_primitivetype
    primitivetype = GeometryType.POLYGON.to_primitivetype
    assert primitivetype is PrimitiveType.POLYGON
    primitivetype = GeometryType.MULTIPOLYGON.to_primitivetype
    assert primitivetype is PrimitiveType.POLYGON

    # A geometry collection doesn't have a primitive type
    with pytest.raises(
        Exception, match="Geometrycollection doesn't have a primitive type"
    ):
        GeometryType.GEOMETRYCOLLECTION.to_primitivetype


def test_primitivetype():
    primitivetype = PrimitiveType(3)
    assert primitivetype is PrimitiveType.POLYGON
    primitivetype = PrimitiveType("PoLyGoN")
    assert primitivetype is PrimitiveType.POLYGON
    primitivetype = PrimitiveType(PrimitiveType.POLYGON)
    assert primitivetype is PrimitiveType.POLYGON


def test_makevalid():
    # Test Point
    point_valid = geometry_util.make_valid(test_helper.TestData.point)
    assert isinstance(point_valid, sh_geom.Point)

    # Test MultiPoint
    multipoint_valid = geometry_util.make_valid(test_helper.TestData.multipoint)
    assert isinstance(multipoint_valid, sh_geom.MultiPoint)

    # Test LineString
    linestring_valid = geometry_util.make_valid(test_helper.TestData.linestring)
    assert isinstance(linestring_valid, sh_geom.LineString)

    # Test MultiLineString
    multilinestring_valid = geometry_util.make_valid(
        test_helper.TestData.multilinestring
    )
    assert isinstance(multilinestring_valid, sh_geom.MultiLineString)

    # Test Polygon, self-intersecting
    polygon_invalid = sh_geom.Polygon(
        shell=[(0, 0), (0, 10), (5, 10), (4, 11), (4, 9), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]],
    )
    poly_valid = geometry_util.make_valid(polygon_invalid)
    assert isinstance(poly_valid, sh_geom.MultiPolygon)
    assert len(poly_valid.geoms) == 2
    assert len(poly_valid.geoms[0].interiors) == 1

    # Test MultiPolygon
    multipolygon_invalid = sh_geom.MultiPolygon(
        [polygon_invalid, test_helper.TestData.polygon_no_islands]
    )
    multipoly_valid = geometry_util.make_valid(multipolygon_invalid)
    assert isinstance(multipoly_valid, sh_geom.MultiPolygon)

    # Test GeometryCollection (as combination of all previous ones)
    geometrycollection_invalid = sh_geom.GeometryCollection(
        [
            test_helper.TestData.point,
            test_helper.TestData.multipoint,
            test_helper.TestData.linestring,
            test_helper.TestData.multilinestring,
            polygon_invalid,
            multipolygon_invalid,
        ]
    )
    geometrycollection_valid = geometry_util.make_valid(geometrycollection_invalid)
    assert isinstance(geometrycollection_valid, sh_geom.GeometryCollection)


def test_numberpoints():
    # Test Point
    numberpoints = geometry_util.numberpoints(test_helper.TestData.point)
    numberpoints_geometrycollection = numberpoints
    assert numberpoints == 1

    # Test MultiPoint
    numberpoints = geometry_util.numberpoints(test_helper.TestData.multipoint)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 3

    # Test LineString
    numberpoints = geometry_util.numberpoints(test_helper.TestData.linestring)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 3

    # Test MultiLineString
    numberpoints = geometry_util.numberpoints(test_helper.TestData.multilinestring)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 6

    # Test Polygon
    numberpoints = geometry_util.numberpoints(test_helper.TestData.polygon_with_island)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 11

    # Test MultiPolygon
    numberpoints = geometry_util.numberpoints(test_helper.TestData.multipolygon)
    numberpoints_geometrycollection += numberpoints
    assert numberpoints == 16

    # Test GeometryCollection (as combination of all previous ones)
    numberpoints = geometry_util.numberpoints(test_helper.TestData.geometrycollection)
    assert numberpoints == numberpoints_geometrycollection


def test_remove_inner_rings():
    # Apply to single Polygon, with area tolerance smaller than holes
    polygon_removerings_withholes = sh_geom.Polygon(
        shell=[(0, 0), (0, 10), (1, 10), (10, 10), (10, 0), (0, 0)],
        holes=[
            [(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)],
            [(5, 5), (5, 6), (7, 6), (7, 5), (5, 5)],
        ],
    )
    poly_result = geometry_util.remove_inner_rings(
        polygon_removerings_withholes, min_area_to_keep=1, crs=None
    )
    assert isinstance(poly_result, sh_geom.Polygon)
    assert len(poly_result.interiors) == 2

    # Apply to single Polygon, with area tolerance between
    # smallest hole (= 2m²) and largest (= 4m²)
    poly_result = geometry_util.remove_inner_rings(
        polygon_removerings_withholes, min_area_to_keep=3, crs=None
    )
    assert isinstance(poly_result, sh_geom.Polygon)
    assert len(poly_result.interiors) == 1

    # Apply to single polygon and remove all holes
    poly_result = geometry_util.remove_inner_rings(
        polygon_removerings_withholes, min_area_to_keep=0, crs=None
    )
    assert isinstance(poly_result, sh_geom.Polygon)
    assert len(poly_result.interiors) == 0
    polygon_removerings_noholes = sh_geom.Polygon(
        shell=[(100, 100), (100, 110), (110, 110), (110, 100), (100, 100)]
    )
    poly_result = geometry_util.remove_inner_rings(
        polygon_removerings_noholes, min_area_to_keep=0, crs=None
    )
    assert isinstance(poly_result, sh_geom.Polygon)
    assert len(poly_result.interiors) == 0

    # Apply to MultiPolygon, with area tolerance between
    # smallest hole (= 2m²) and largest (= 4m²)
    multipoly_removerings = sh_geom.MultiPolygon(
        [polygon_removerings_withholes, polygon_removerings_noholes]
    )
    poly_result = geometry_util.remove_inner_rings(
        multipoly_removerings, min_area_to_keep=3, crs=None
    )
    assert isinstance(poly_result, sh_geom.MultiPolygon)
    assert len(poly_result.geoms[0].interiors) == 1


def test_simplify_coords_lang():
    # Test LineString, lookahead -1, via coordinates
    linestring = sh_geom.LineString([(0, 0), (10, 10), (20, 20)])
    coords_simplified = geometry_util.simplify_coords_lang(
        coords=linestring.coords, tolerance=1, lookahead=-1
    )
    assert isinstance(coords_simplified, list)
    assert len(coords_simplified) < len(linestring.coords)
    assert len(coords_simplified) == 2


def test_simplify_ext_lang_basic():
    # Test LineString, lookahead -1, via geometry
    linestring = sh_geom.LineString([(0, 0), (10, 10), (20, 20)])
    geom_simplified = geometry_util.simplify_ext(
        geometry=linestring,
        algorithm=geometry_util.SimplifyAlgorithm.LANG,
        tolerance=1,
        lookahead=-1,
    )
    assert isinstance(geom_simplified, sh_geom.LineString)
    assert len(geom_simplified.coords) < len(linestring.coords)
    assert len(geom_simplified.coords) == 2

    # Test Polygon lookahead -1
    poly = sh_geom.Polygon(
        shell=[(0, 0), (0, 10), (1, 10), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]],
    )
    geom_simplified = geometry_util.simplify_ext(
        geometry=poly,
        algorithm=geometry_util.SimplifyAlgorithm.LANG,
        tolerance=1,
        lookahead=-1,
    )
    assert isinstance(geom_simplified, sh_geom.Polygon)
    assert geom_simplified.exterior is not None
    assert poly.exterior is not None
    assert len(geom_simplified.exterior.coords) < len(poly.exterior.coords)
    assert len(geom_simplified.exterior.coords) == 5

    # Test Point simplification
    point = sh_geom.Point((0, 0))
    geom_simplified = geometry_util.simplify_ext(
        geometry=point, algorithm=geometry_util.SimplifyAlgorithm.LANG, tolerance=1
    )
    assert isinstance(geom_simplified, sh_geom.Point)
    assert len(geom_simplified.coords) == 1

    # Test MultiPoint simplification
    multipoint = sh_geom.MultiPoint([(0, 0), (10, 10), (20, 20)])
    geom_simplified = geometry_util.simplify_ext(
        geometry=multipoint, algorithm=geometry_util.SimplifyAlgorithm.LANG, tolerance=1
    )
    assert isinstance(geom_simplified, sh_geom.MultiPoint)
    assert len(geom_simplified.geoms) == 3

    # Test LineString simplification
    linestring = sh_geom.LineString([(0, 0), (10, 10), (20, 20)])
    geom_simplified = geometry_util.simplify_ext(
        geometry=linestring, algorithm=geometry_util.SimplifyAlgorithm.LANG, tolerance=1
    )
    assert isinstance(geom_simplified, sh_geom.LineString)
    assert len(geom_simplified.coords) < len(linestring.coords)
    assert len(geom_simplified.coords) == 2

    # Test MultiLineString simplification
    multilinestring = sh_geom.MultiLineString(
        [list(linestring.coords), [(100, 100), (110, 110), (120, 120)]]
    )
    geom_simplified = geometry_util.simplify_ext(
        geometry=multilinestring,
        algorithm=geometry_util.SimplifyAlgorithm.LANG,
        tolerance=1,
    )
    assert isinstance(geom_simplified, sh_geom.MultiLineString)
    assert len(geom_simplified.geoms) == 2
    assert len(geom_simplified.geoms[0].coords) < len(multilinestring.geoms[0].coords)
    assert len(geom_simplified.geoms[0].coords) == 2

    # Test Polygon simplification
    poly = sh_geom.Polygon(
        shell=[(0, 0), (0, 10), (1, 10), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]],
    )
    geom_simplified = geometry_util.simplify_ext(
        geometry=poly, algorithm=geometry_util.SimplifyAlgorithm.LANG, tolerance=1
    )
    assert isinstance(geom_simplified, sh_geom.Polygon)
    assert geom_simplified.exterior is not None
    assert poly.exterior is not None
    assert len(geom_simplified.exterior.coords) < len(poly.exterior.coords)
    assert len(geom_simplified.exterior.coords) == 5

    # Test MultiPolygon simplification
    poly2 = sh_geom.Polygon(
        shell=[(100, 100), (100, 110), (110, 110), (110, 100), (100, 100)]
    )
    multipoly = sh_geom.MultiPolygon([poly, poly2])
    geom_simplified = geometry_util.simplify_ext(
        geometry=multipoly, algorithm=geometry_util.SimplifyAlgorithm.LANG, tolerance=1
    )
    assert isinstance(geom_simplified, sh_geom.MultiPolygon)
    assert len(geom_simplified.geoms) == 2
    assert len(geom_simplified.geoms[0].exterior.coords) < len(poly.exterior.coords)
    assert len(geom_simplified.geoms[0].exterior.coords) == 5

    # Test GeometryCollection (as combination of all previous ones) simplification
    geom = sh_geom.GeometryCollection(
        [point, multipoint, linestring, multilinestring, poly, multipoly]
    )
    geom_simplified = geometry_util.simplify_ext(
        geometry=geom, algorithm=geometry_util.SimplifyAlgorithm.LANG, tolerance=1
    )
    assert isinstance(geom_simplified, sh_geom.GeometryCollection)
    assert len(geom_simplified.geoms) == 6


def test_simplify_ext_lang_preservetopology():
    # Test Polygon lookahead -1
    poly = sh_geom.Polygon(
        shell=[(0, 0), (0, 10), (1, 10), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]],
    )
    # If preserve_topology True, the original polygon is returned...
    geom_simplified = geometry_util.simplify_ext(
        geometry=poly,
        algorithm=geometry_util.SimplifyAlgorithm.LANG,
        tolerance=10,
        preserve_topology=True,
        lookahead=-1,
    )
    assert isinstance(geom_simplified, sh_geom.Polygon)
    assert poly.equals(geom_simplified) is True

    # If preserve_topology True, the original polygon is returned...
    geom_simplified = geometry_util.simplify_ext(
        geometry=poly,
        algorithm=geometry_util.SimplifyAlgorithm.LANG,
        tolerance=10,
        preserve_topology=False,
        lookahead=-1,
    )
    assert geom_simplified is None


def test_simplify_ext_invalid():
    # Test Polygon simplification, with invalid exterior ring
    poly = sh_geom.Polygon(
        shell=[(0, 0), (0, 10), (5, 10), (3, 12), (3, 9), (10, 10), (10, 0), (0, 0)],
        holes=[[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]],
    )
    geom_simplified = geometry_util.simplify_ext(
        geometry=poly, algorithm=geometry_util.SimplifyAlgorithm.LANG, tolerance=1
    )
    assert isinstance(geom_simplified, sh_geom.MultiPolygon)
    assert poly.exterior is not None
    assert len(geom_simplified.geoms[0].exterior.coords) < len(poly.exterior.coords)
    assert len(geom_simplified.geoms[0].exterior.coords) == 7
    assert len(geom_simplified.geoms[0].interiors) == len(poly.interiors)

    # Test Polygon simplification, with exterior ring that touches itself
    # due to simplification and after make_valid results in multipolygon of
    # 2 equally large parts (left and right part of M shape).
    poly_m_touch = sh_geom.Polygon(
        shell=[
            (0, 0),
            (0, 10),
            (5, 5),
            (10, 10),
            (10, 0),
            (8, 0),
            (8, 5),
            (5, 4),
            (2, 5),
            (2, 0),
            (0, 0),
        ]
    )
    geom_simplified = geometry_util.simplify_ext(
        geometry=poly_m_touch,
        algorithm=geometry_util.SimplifyAlgorithm.LANG,
        tolerance=1,
    )
    assert geom_simplified is not None
    assert geom_simplified.is_valid
    assert isinstance(geom_simplified, sh_geom.MultiPolygon)
    assert len(geom_simplified.geoms) == 2
    assert geometry_util.numberpoints(geom_simplified) < geometry_util.numberpoints(
        poly
    )

    # Test Polygon simplification, with exterior ring that crosses itself
    # due to simplification and after make_valid results in multipolygon of
    # 3 parts (left, middle and right part of M shape).
    poly_m_cross = sh_geom.Polygon(
        shell=[
            (0, 0),
            (0, 10),
            (5, 5),
            (10, 10),
            (10, 0),
            (8, 0),
            (8, 5.5),
            (5, 4.5),
            (2, 5.5),
            (2, 0),
            (0, 0),
        ]
    )
    geom_simplified = geometry_util.simplify_ext(
        geometry=poly_m_cross,
        algorithm=geometry_util.SimplifyAlgorithm.LANG,
        tolerance=1,
    )
    assert geom_simplified is not None
    assert geom_simplified.is_valid
    assert isinstance(geom_simplified, sh_geom.MultiPolygon)
    assert len(geom_simplified.geoms) == 3


def test_simplify_ext_keep_points_on(tmp_path):
    # First init some stuff
    input_path = test_helper.get_testfile("polygon-simplify-onborder-testcase")
    input_gdf = gfo.read_file(input_path)

    # Create geometry where we want the points kept
    grid_gdf = grid_util.create_grid(
        total_bounds=(
            210431.875 - 1000,
            176640.125 - 1000,
            210431.875 + 1000,
            176640.125 + 1000,
        ),
        nb_columns=2,
        nb_rows=2,
        crs="epsg:31370",
    )
    gfo.to_file(grid_gdf, tmp_path / "grid.gpkg")
    grid_coords = [tile.exterior.coords for tile in grid_gdf["geometry"]]
    grid_lines_geom = sh_geom.MultiLineString(grid_coords)

    # Test rdp (ramer–douglas–peucker)
    # Without keep_points_on, the following point that is on the test data +
    # on the grid is removed by rdp
    point_on_input_and_border = sh_geom.Point(210431.875, 176599.375)
    tolerance_rdp = 0.5

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(
        input_gdf[input_gdf.intersects(point_on_input_and_border)]
    )
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes
    simplified_gdf = input_gdf.copy()
    # assert to evade pyLance warning
    assert isinstance(simplified_gdf, gpd.GeoDataFrame)
    simplified_gdf.geometry = input_gdf.geometry.apply(
        lambda geom: geometry_util.simplify_ext(
            geom,
            algorithm=geometry_util.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
            tolerance=tolerance_rdp,
        )
    )
    gfo.to_file(simplified_gdf, tmp_path / f"simplified_rdp{tolerance_rdp}.gpkg")
    assert (
        len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)])
        != nb_intersects_with_input
    )

    # With keep_points_on specified, the number of intersections stays the same
    simplified_gdf = input_gdf.copy()
    # assert to evade pyLance warning
    assert isinstance(simplified_gdf, gpd.GeoDataFrame)
    simplified_gdf.geometry = input_gdf.geometry.apply(
        lambda geom: geometry_util.simplify_ext(
            geom,
            algorithm=geometry_util.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
            tolerance=tolerance_rdp,
            keep_points_on=grid_lines_geom,
        )
    )
    gfo.to_file(
        simplified_gdf, tmp_path / f"simplified_rdp{tolerance_rdp}_keep_points_on.gpkg"
    )
    assert (
        len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)])
        == nb_intersects_with_input
    )

    # Test vw (visvalingam-whyatt)
    # Without keep_points_on, the following point that is on the test data +
    # on the grid is removed by vw
    point_on_input_and_border = sh_geom.Point(210430.125, 176640.125)
    tolerance_vw = 16 * 0.25 * 0.25  # 1m²

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(
        input_gdf[input_gdf.intersects(point_on_input_and_border)]
    )
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes
    simplified_gdf = input_gdf.copy()
    # assert to evade pyLance warning
    assert isinstance(simplified_gdf, gpd.GeoDataFrame)
    simplified_gdf.geometry = input_gdf.geometry.apply(
        lambda geom: geometry_util.simplify_ext(
            geom,
            algorithm=geometry_util.SimplifyAlgorithm.VISVALINGAM_WHYATT,
            tolerance=tolerance_vw,
        )
    )
    gfo.to_file(simplified_gdf, tmp_path / f"simplified_vw{tolerance_vw}.gpkg")
    assert (
        len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)])
        != nb_intersects_with_input
    )

    # With keep_points_on specified, the number of intersections stays the same
    simplified_gdf = input_gdf.copy()
    # assert to evade pyLance warning
    assert isinstance(simplified_gdf, gpd.GeoDataFrame)
    simplified_gdf.geometry = input_gdf.geometry.apply(
        lambda geom: geometry_util.simplify_ext(
            geom,
            algorithm=geometry_util.SimplifyAlgorithm.VISVALINGAM_WHYATT,
            tolerance=tolerance_vw,
            keep_points_on=grid_lines_geom,
        )
    )
    gfo.to_file(
        simplified_gdf, tmp_path / f"simplified_vw{tolerance_vw}_keep_points_on.gpkg"
    )
    assert (
        len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)])
        == nb_intersects_with_input
    )

    # Test lang
    # Without keep_points_on, the following point that is on the test data +
    # on the grid is removed by lang
    point_on_input_and_border = sh_geom.Point(210431.875, 176606.125)
    tolerance_lang = 0.25
    step_lang = 8

    # Determine the number of intersects with the input test data
    nb_intersects_with_input = len(
        input_gdf[input_gdf.intersects(point_on_input_and_border)]
    )
    assert nb_intersects_with_input > 0
    # Test if intersects > 0
    assert len(input_gdf[grid_gdf.intersects(point_on_input_and_border)]) > 0

    # Without keep_points_on the number of intersections changes
    simplified_gdf = input_gdf.copy()
    # assert to evade pyLance warning
    assert isinstance(simplified_gdf, gpd.GeoDataFrame)
    simplified_gdf.geometry = input_gdf.geometry.apply(
        lambda geom: geometry_util.simplify_ext(
            geom,
            algorithm=geometry_util.SimplifyAlgorithm.LANG,
            tolerance=tolerance_lang,
            lookahead=step_lang,
        )
    )
    gfo.to_file(
        simplified_gdf, tmp_path / f"simplified_lang;{tolerance_lang};{step_lang}.gpkg"
    )
    assert (
        len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)])
        != nb_intersects_with_input
    )

    # With keep_points_on specified, the number of intersections stays the same
    simplified_gdf = input_gdf.copy()
    # assert to evade pyLance warning
    assert isinstance(simplified_gdf, gpd.GeoDataFrame)
    simplified_gdf.geometry = input_gdf.geometry.apply(
        lambda geom: geometry_util.simplify_ext(
            geom,
            algorithm=geometry_util.SimplifyAlgorithm.LANG,
            tolerance=tolerance_lang,
            lookahead=step_lang,
            keep_points_on=grid_lines_geom,
        )
    )
    output_path = (
        tmp_path / f"simplified_lang;{tolerance_lang};{step_lang}_keep_points_on.gpkg"
    )
    gfo.to_file(simplified_gdf, output_path)
    assert (
        len(simplified_gdf[simplified_gdf.intersects(point_on_input_and_border)])
        == nb_intersects_with_input
    )


def test_simplify_ext_no_simplification():
    # Backup reference to simplification module
    _temp_simplification = None
    if sys.modules.get("simplification"):
        _temp_simplification = sys.modules["simplification"]
    try:
        # Fake that the module is not available
        sys.modules["simplification"] = None  # type: ignore

        # Using RDP needs simplification module, so should give ImportError
        geometry_util.simplify_ext(
            geometry=sh_geom.LineString([(0, 0), (10, 10), (20, 20)]),
            algorithm=geometry_util.SimplifyAlgorithm.RAMER_DOUGLAS_PEUCKER,
            tolerance=1,
        )
        assert True is False
    except ImportError:
        assert True is True
    finally:
        if _temp_simplification:
            sys.modules["simplification"] = _temp_simplification
        else:
            del sys.modules["simplification"]


@pytest.mark.parametrize(
    "descr, exp_angle_start, exp_angle_end, wkt",
    [
        ["Geom EMPTY", np.nan, np.nan, "POLYGON(EMPTY)"],
        ["Geom None", np.nan, np.nan, None],
        [
            "NE>NW_<360",
            45.0,
            135.0,
            "POLYGON((1 1, 1 -1, -1 -1, -1 1, -1 -1, -2 -2, 2 -2, 1 1))",
        ],
        [
            "NE>SW_<360",
            0.0,
            270.0,
            "POLYGON((1 0, 0 1, -1 0, 0 -1, -1 -1, -2 2, 2 2, 1 0))",
        ],
        [
            "NE>SW_360",
            0.0,
            360.0,
            "POLYGON((1 0, 1 1, -1 1, -1 -1, 1 -1, 2 0, 2 -2, -2 -2, -2 2, 2 2, 1 0))",
        ],
        ["NE_SE", 315.0, 45.0, "POLYGON((1 1, 1 -1, 2 -1, 2 1, 1 1))"],
        ["NE, y=0", 0.0, 45.0, "POLYGON((1 0, 1 1, 2 1, 2 0, 1 0))"],
        ["NW", 135.0, 180.0, "POLYGON((-1 0, -1 1, -2 1, -2 0, -1 0))"],
        ["NW_SE", 135.0, 315.0, "POLYGON((-1 1, -1 0.5, 1 -1, -3 1, -1 1))"],
        ["NW_SW", 135.0, 225.0, "POLYGON((-1 -1, -1 1, -2 1, -2 -1, -1 -1))"],
        [
            "NW>SE",
            135.0,
            360.0,
            "POLYGON((-1 1, -1 -1, 1 -1, 1 0, 2 -2, -2 -2, -1 1))",
        ],
        ["SE, y!=0", 270.0, 315.0, "POLYGON((1 -1, 0 -1, 0 -2, 1 -1))"],
        ["SE, y=0", 315.0, 360.0, "POLYGON((1 0, 1 -1, 2 -1, 2 0, 1 0))"],
        [
            "SW>NW",
            225.0,
            135.0,
            "POLYGON((-1 -1, 2 -1, -1 1, 3 1, 3 -2, -1 -1))",
        ],
    ],
)
def test_view_angles(descr, exp_angle_start, exp_angle_end, wkt):
    # View location
    viewpoint_x = 10
    viewpoint_y = 20
    viewpoint = pygeos.from_wkt(f"POINT({viewpoint_x} {viewpoint_y})")
    visible_geom_orig = pygeos.from_wkt(wkt)

    # The raw visible geoms aree based on a view location of 0,0. Adapt it to the
    # view location used as 0,0 wouldn't have a good test coverage.
    def add_viewpoint(coords):
        return np.column_stack([coords[:, 0] + viewpoint_x, coords[:, 1] + viewpoint_y])

    visible_geom = pygeos.apply(visible_geom_orig, add_viewpoint)

    # Go!
    view_angles = geometry_util.view_angles(viewpoint, visible_geom)
    assert view_angles == (exp_angle_start, exp_angle_end)
