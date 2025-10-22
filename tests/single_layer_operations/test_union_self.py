import math
from itertools import product

import geopandas as gpd
import pytest
from shapely import box

import geofileops as gfo
from geofileops import GeometryType
from geofileops.util._geofileinfo import GeofileInfo
from tests import test_helper
from tests.test_helper import SUFFIXES_GEOOPS


@pytest.mark.parametrize("suffix", SUFFIXES_GEOOPS)
def test_union_full_self_3circles(tmp_path, suffix: str):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-3overlappingcircles", suffix=suffix)
    input_layerinfo = gfo.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"

    # Also run some tests on basic data with circles
    # Union the single circle towards the 2 circles
    gfo.union_full_self(
        input_path=input_path,
        output_path=output_path,
        batchsize=batchsize,
    )

    # Check if the tmp file is correctly created
    assert output_path.exists()
    exp_spatial_index = GeofileInfo(output_path).default_spatial_index
    assert gfo.has_spatial_index(output_path) is exp_spatial_index
    output_layerinfo = gfo.get_layerinfo(output_path)
    assert output_layerinfo.featurecount == 7
    assert ((len(input_layerinfo.columns) + 1) * 3) == len(output_layerinfo.columns)
    assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None


@pytest.mark.parametrize("suffix", SUFFIXES_GEOOPS)
def test_union_full_self_4circles(tmp_path, suffix: str):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-4overlappingcircles", suffix=suffix)
    input_layerinfo = gfo.get_layerinfo(input_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    output_path = tmp_path / f"{input_path.stem}-output{suffix}"

    # Also run some tests on basic data with circles
    # Union the single circle towards the 2 circles
    gfo.union_full_self(
        input_path=input_path,
        output_path=output_path,
        batchsize=batchsize,
    )

    # Check if the tmp file is correctly created
    assert output_path.exists()
    exp_spatial_index = GeofileInfo(output_path).default_spatial_index
    assert gfo.has_spatial_index(output_path) is exp_spatial_index
    output_layerinfo = gfo.get_layerinfo(output_path)
    assert output_layerinfo.featurecount == 11
    assert ((len(input_layerinfo.columns) + 1) * 4) == len(output_layerinfo.columns)
    assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None


@pytest.mark.parametrize("suffix", [".gpkg"])
@pytest.mark.parametrize(
    "nb_boxes, union_type, columns, exp_features",
    [
        (2, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", None, 3),
        (2, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", ["name"], 3),
        (2, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", ["fid", "value", "name"], 3),
        (3, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", None, 7),
        (4, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", None, 9),
        (5, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", ["fid", "value", "name"], 17),
        (2, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", [], 3),
        (3, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", [], 7),
        (4, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", [], 9),
        (5, "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS", [], 17),
        (2, "NO_INTERSECTIONS_ATTRIBUTE_LISTS", None, 3),
        (3, "NO_INTERSECTIONS_ATTRIBUTE_LISTS", None, 7),
        (4, "NO_INTERSECTIONS_ATTRIBUTE_LISTS", None, 9),
        (5, "NO_INTERSECTIONS_ATTRIBUTE_LISTS", ["fid", "value", "name"], 17),
        (2, "REPEATED_INTERSECTIONS", None, 4),
        (3, "REPEATED_INTERSECTIONS", None, 12),
        (4, "REPEATED_INTERSECTIONS", None, 16),
        (5, "REPEATED_INTERSECTIONS", ["fid", "value", "name"], 37),
    ],
)
def test_union_full_self_boxes(
    tmp_path, suffix, nb_boxes, union_type, columns, exp_features
):
    # Prepare test data
    if nb_boxes == 2:
        boxes = [box(0, 0, 10, 10), box(8, 0, 18, 10)]
    elif nb_boxes == 3:
        boxes = [box(0, 0, 10, 10), box(8, 0, 18, 10), box(5, 8, 15, 18)]
    elif nb_boxes == 4:
        boxes = [
            box(0, 0, 10, 10),
            box(8, 0, 18, 10),
            box(0, 8, 10, 18),
            box(8, 8, 18, 18),
        ]
    elif nb_boxes == 5:
        boxes = [
            box(0, 0, 10, 10),
            box(8, 0, 18, 10),
            box(0, 8, 10, 18),
            box(8, 8, 18, 18),
            box(5, 5, 15, 15),
        ]
    else:
        raise ValueError(f"Unsupported number of boxes for this test {nb_boxes}.")

    batchsize = math.ceil(nb_boxes / 2)
    input_gdf = gpd.GeoDataFrame(
        {
            "geometry": boxes,
            "value": range(len(boxes)),
            "name": [f"box_{i}" for i in range(len(boxes))],
        },
        crs="EPSG:31370",
    )
    input_path = tmp_path / f"input_boxes-{nb_boxes}{suffix}"
    input_gdf.to_file(input_path)

    # Run test
    output_path = tmp_path / f"output_boxes-{nb_boxes}_{union_type}{suffix}"
    gfo.union_full_self(
        input_path=input_path,
        output_path=output_path,
        union_type=union_type,
        columns=columns,
        batchsize=batchsize,
    )

    # Check if the output file is correctly created
    input_layerinfo = gfo.get_layerinfo(input_path)
    asked_columns = columns if columns is not None else list(input_layerinfo.columns)
    if union_type == "NO_INTERSECTIONS_ATTRIBUTE_COLUMNS":
        # asked columns * nb_intersections (=boxes)
        exp_columns = [
            f"is{idx}_{col}" for idx, col in product(range(len(boxes)), asked_columns)
        ]
        # The "nb_intersections" column is not available in this union_type
        exp_max_nb_intersections = None
    elif union_type == "NO_INTERSECTIONS_ATTRIBUTE_LISTS":
        # asked columns + "nb_intersections"
        exp_columns = [col if col != "fid" else "fid_1" for col in asked_columns]
        exp_columns += ["nb_intersections"]
        # The "nb_intersections" column is filled out for this union_type
        exp_max_nb_intersections = nb_boxes - 1
    elif union_type == "REPEATED_INTERSECTIONS":
        # asked columns + "union_fid"
        exp_columns = [col if col != "fid" else "fid_1" for col in asked_columns]
        exp_columns += ["union_fid"]
        # The "nb_intersections" column is not available in this union_type
        exp_max_nb_intersections = None
    else:
        raise ValueError(f"Unsupported union_type for this test: {union_type}")

    assert output_path.exists()
    exp_spatial_index = GeofileInfo(output_path).default_spatial_index
    assert gfo.has_spatial_index(output_path) is exp_spatial_index
    output_layerinfo = gfo.get_layerinfo(output_path)
    assert output_layerinfo.featurecount == exp_features
    assert sorted(output_layerinfo.columns) == sorted(exp_columns)
    assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None
    if exp_max_nb_intersections is not None:
        assert "nb_intersections" in output_gdf.columns
        assert output_gdf["nb_intersections"].max().item() == exp_max_nb_intersections
    else:
        assert "nb_intersections" not in output_gdf.columns


@pytest.mark.parametrize(
    "kwargs, error_msg",
    [
        ({"subdivide_coords": -5}, "subdivide_coords < 0 is not allowed"),
        ({"union_type": "INVALID_TYPE"}, "union_type should be one of"),
        (
            {"union_type": "NO_INTERSECTIONS_NO_ATTRIBUTES", "columns": ["name"]},
            "input_columns should not be set when union_type is "
            "'NO_INTERSECTIONS_NO_ATTRIBUTES'",
        ),
    ],
)
def test_union_full_self_invalid_args(tmp_path, kwargs, error_msg):
    # Prepare test data
    input_path = test_helper.get_testfile("polygon-3overlappingcircles", suffix=".gpkg")
    output_path = tmp_path / "output_invalid_union_type.gpkg"

    with pytest.raises(ValueError, match=error_msg):
        gfo.union_full_self(
            input_path=input_path,
            output_path=output_path,
            **kwargs,
        )
