import math

import geopandas as gpd
import pytest
import shapely

import geofileops as gfo
from geofileops import GeometryType
from geofileops.util._geofileinfo import GeofileInfo
from tests import test_helper
from tests.test_helper import SUFFIXES_GEOOPS

zones = [
    "POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0))",
    "POLYGON ((10 1, 13 1, 13 4, 10 4, 10 1))",
    "POLYGON ((7 2, 8 2, 8 5, 1 5, 1 4, 7 4, 7 2))",
    "POLYGON ((7 1, 7 -2, 10 -2, 10 -1, 8 -1, 8 1, 7 1))",
]

extra_parcels = [
    "POLYGON ((-2 4, 0 4, 0 6, -2 6, -2 4))",
    "POLYGON ((3 6, 5 6, 5 8, 3 8, 3 6))",
]


@pytest.mark.parametrize("suffix", SUFFIXES_GEOOPS)
@pytest.mark.parametrize(
    "columns, gridsize, where_post, subdivide_coords, exp_featurecount",
    [
        (["OIDN", "UIDN"], 0.0, "ST_Area(geom) > 2000", 0, 25),
        (None, 0.01, None, 10, 27),
    ],
)
def test_export_by_location(
    tmp_path,
    suffix,
    columns,
    gridsize,
    where_post,
    subdivide_coords,
    exp_featurecount,
):
    input_to_select_from_path = test_helper.get_testfile(
        "polygon-parcel", suffix=suffix
    )
    input_to_compare_with_path = test_helper.get_testfile("polygon-zone", suffix=suffix)
    output_path = tmp_path / f"{input_to_select_from_path.stem}-output{suffix}"
    input_layerinfo = gfo.get_layerinfo(input_to_select_from_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)

    # Test
    gfo.export_by_location(
        input_to_select_from_path=str(input_to_select_from_path),
        input_to_compare_with_path=str(input_to_compare_with_path),
        output_path=str(output_path),
        input1_columns=columns,
        gridsize=gridsize,
        where_post=where_post,
        subdivide_coords=subdivide_coords,
        batchsize=batchsize,
    )

    # Check if the output file is correctly created
    assert output_path.exists()
    exp_spatial_index = GeofileInfo(output_path).default_spatial_index
    assert gfo.has_spatial_index(output_path) is exp_spatial_index
    output_layerinfo = gfo.get_layerinfo(output_path)
    exp_columns = len(input_layerinfo.columns) if columns is None else len(columns)
    assert len(output_layerinfo.columns) == exp_columns
    assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON
    assert output_layerinfo.featurecount == exp_featurecount

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None


@pytest.mark.parametrize(
    "query, area_inters_column_name, min_area_intersect, subdivide_coords, "
    "exp_featurecount",
    [
        (None, None, None, 10, 27),
        (None, "area_custom", None, 0, 27),
        ("within is False", "area_custom", None, 0, 39),
        (None, None, 1000, 10, 24),
        ("within is False", None, 1000, 0, 15),
    ],
)
@pytest.mark.filterwarnings("ignore:.*Field format '' not supported.*")
def test_export_by_location_area(
    tmp_path,
    query,
    area_inters_column_name,
    min_area_intersect,
    subdivide_coords,
    exp_featurecount,
):
    input_to_select_from_path = test_helper.get_testfile("polygon-parcel")
    input_to_compare_with_path = test_helper.get_testfile("polygon-zone")
    output_path = tmp_path / f"{input_to_select_from_path.stem}-output.gpkg"
    input_layerinfo = gfo.get_layerinfo(input_to_select_from_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)

    # Test
    kwargs = {}
    if query is not None:
        kwargs["spatial_relations_query"] = query
    gfo.export_by_location(
        input_to_select_from_path=str(input_to_select_from_path),
        input_to_compare_with_path=str(input_to_compare_with_path),
        output_path=str(output_path),
        area_inters_column_name=area_inters_column_name,
        min_area_intersect=min_area_intersect,
        batchsize=batchsize,
        subdivide_coords=subdivide_coords,
        **kwargs,
    )

    # Check if the output file is correctly created
    assert output_path.exists()
    exp_spatial_index = GeofileInfo(output_path).default_spatial_index
    assert gfo.has_spatial_index(output_path) is exp_spatial_index
    output_layerinfo = gfo.get_layerinfo(output_path)
    exp_columns = len(input_layerinfo.columns)
    exp_area_inters_column_name = area_inters_column_name
    if exp_area_inters_column_name is None and min_area_intersect is not None:
        exp_area_inters_column_name = "area_inters"
    if exp_area_inters_column_name is not None:
        exp_columns += 1
        assert exp_area_inters_column_name in output_layerinfo.columns
    assert len(output_layerinfo.columns) == exp_columns
    assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON
    assert output_layerinfo.featurecount == exp_featurecount

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None

    # If an area column name is specified, check the number of None values
    # (= number features without intersection)
    if area_inters_column_name is not None:
        exp_nb_None = 21 if query == "within is False" else 0
        assert len(output_gdf[output_gdf.area_custom.isna()]) == exp_nb_None


def test_export_by_location_force(tmp_path):
    # Prepare test data
    input1_path = test_helper.get_testfile("polygon-parcel")
    input2_path = test_helper.get_testfile("polygon-zone")
    output_path = tmp_path / f"output{input1_path.suffix}"
    output_path.touch()

    # Test with force False (the default): existing output file should stay the same
    mtime_orig = output_path.stat().st_mtime
    gfo.export_by_location(
        input_to_select_from_path=input1_path,
        input_to_compare_with_path=input2_path,
        output_path=output_path,
    )
    assert output_path.stat().st_mtime == mtime_orig

    # With force=True
    gfo.export_by_location(
        input_to_select_from_path=input1_path,
        input_to_compare_with_path=input2_path,
        output_path=output_path,
        force=True,
    )
    assert output_path.stat().st_mtime != mtime_orig


@pytest.mark.parametrize(
    "kwargs, expected_error",
    [
        ({"subdivide_coords": -1}, "subdivide_coords < 0 is not allowed"),
    ],
)
def test_export_by_location_invalid_params(kwargs, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        gfo.export_by_location(
            input_to_select_from_path="input.gpkg",
            input_to_compare_with_path="input2.gpkg",
            output_path="output.gpkg",
            **kwargs,
        )


@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "query, exp_featurecount",
    [
        ("intersects is True or intersects is False", 48),
        ("intersects is True", 27),
        ("intersects is False", 21),
        ("within is True", 8),
        ("within is False", 40),
        ("disjoint is True", 21),
        ("disjoint is False", 27),
        ("equals is True", 0),
        ("equals is False", 48),
        ("coveredby is True", 8),
        ("coveredby is False", 40),
        ("covers is True", 0),
        ("covers is False", 48),
    ],
)
def test_export_by_location_query(tmp_path, query, subdivide_coords, exp_featurecount):
    input_to_select_from_path = test_helper.get_testfile("polygon-parcel")
    input_to_compare_with_path = test_helper.get_testfile("polygon-zone")
    output_path = tmp_path / f"{input_to_select_from_path.stem}-output.gpkg"
    input_layerinfo = gfo.get_layerinfo(input_to_select_from_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    batchsize = input_layerinfo.featurecount

    # Test
    gfo.export_by_location(
        input_to_select_from_path=str(input_to_select_from_path),
        input_to_compare_with_path=str(input_to_compare_with_path),
        output_path=str(output_path),
        spatial_relations_query=query,
        batchsize=batchsize,
        subdivide_coords=subdivide_coords,
    )

    # Check if the output file is correctly created
    assert output_path.exists()
    exp_spatial_index = GeofileInfo(output_path).default_spatial_index
    assert gfo.has_spatial_index(output_path) is exp_spatial_index
    output_layerinfo = gfo.get_layerinfo(output_path)
    exp_columns = len(input_layerinfo.columns)
    assert len(output_layerinfo.columns) == exp_columns
    assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON
    assert output_layerinfo.featurecount == exp_featurecount

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    if exp_featurecount > 0:
        assert output_gdf["geometry"][0] is not None
    elif exp_featurecount == 0:
        assert output_gdf.empty


@pytest.mark.parametrize(
    "query, exp_featurecount",
    [
        ("", 48),
    ],
)
def test_export_by_location_empty_query(tmp_path, query, exp_featurecount):
    input_to_select_from_path = test_helper.get_testfile("polygon-parcel")
    input_to_compare_with_path = test_helper.get_testfile("polygon-zone")
    output_path = tmp_path / f"{input_to_select_from_path.stem}-output.gpkg"
    input_layerinfo = gfo.get_layerinfo(input_to_select_from_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)

    # Test
    gfo.export_by_location(
        input_to_select_from_path=str(input_to_select_from_path),
        input_to_compare_with_path=str(input_to_compare_with_path),
        output_path=str(output_path),
        spatial_relations_query=query,
        batchsize=batchsize,
        area_inters_column_name="area_inters",
    )

    # Check if the output file is correctly created
    assert output_path.exists()
    exp_spatial_index = GeofileInfo(output_path).default_spatial_index
    assert gfo.has_spatial_index(output_path) is exp_spatial_index
    output_layerinfo = gfo.get_layerinfo(output_path)
    exp_columns = len(input_layerinfo.columns) + 1
    assert len(output_layerinfo.columns) == exp_columns
    assert output_layerinfo.geometrytype == GeometryType.MULTIPOLYGON
    assert output_layerinfo.featurecount == exp_featurecount


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels", [["POLYGON ((2 1, 4 1, 4 3, 2 3, 2 1))", *extra_parcels]]
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("intersects is True", 1), ("intersects is False", 2)],
)
def test_intersects(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels", [["POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))", *extra_parcels]]
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("within is True", 1), ("within is False", 2)],
)
def test_within(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels", [["POLYGON ((2 1, 4 1, 4 3, 2 3, 2 1))", *extra_parcels]]
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("overlaps is True", 1), ("overlaps is False", 2)],
)
def test_overlaps(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels", [["POLYGON ((3 1, 5 1, 5 3, 3 3, 3 1))", *extra_parcels]]
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("touches is True", 1), ("touches is False", 2)],
)
def test_touches(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels",
    [["POLYGON ((0.5 0.5, 2.5 0.5, 2.5 2.5, 0.5 2.5, 0.5 0.5))", *extra_parcels]],
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("contains is True", 1), ("contains is False", 2)],
)
def test_contains(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels",
    [["POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0))", *extra_parcels]],
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("equals is True", 1), ("equals is False", 2)],
)
def test_equals(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels",
    [["POLYGON ((-1 -1, 4 -1, 4 4, -1 4, -1 -1))", *extra_parcels]],
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("covers is True", 1), ("covers is False", 2)],
)
def test_covers(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels",
    [["POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))", *extra_parcels]],
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("coveredby is True", 1), ("coveredby is False", 2)],
)
def test_coveredby(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels",
    [
        [
            "POLYGON ((4 1, 6 1, 6 3, 4 3, 4 1))",
            "POLYGON ((4 -2, 6 -2, 6 0, 4 0, 4 -2))",
            "POLYGON ((9 0, 11 0, 11 2, 9 2, 9 0))",
            *extra_parcels,
        ]
    ],
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("disjoint is True", 4), ("disjoint is False", 1)],
)
def test_disjoint(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels",
    [
        [
            "POLYGON ((2 1, 4 1, 4 3, 2 3, 2 1))",
            "POLYGON ((2 -2, 4 -2, 4 0, 2 0, 2 -2))",
            *extra_parcels,
        ]
    ],
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [
        ("intersects is True and touches is False", 1),
        ("intersects is True and touches is True", 1),
        ("intersects is True or touches is False", 4),
        ("intersects is True or touches is True", 2),
    ],
)
def test_intersects_true_touches_false(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    _spatial_relation(
        tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
    )


def _spatial_relation(
    tmp_path, spatial_relations, zones, parcels, exp_features, subdivide_coords
):
    input_to_compare_with_path = tmp_path / "zones.gpkg"
    box_zones = []
    for zone in zones:
        box_zone = shapely.from_wkt(zone)
        box_zone = shapely.segmentize(box_zone, 1)
        box_zones.append(box_zone)
    gdf_zone = gpd.GeoDataFrame(geometry=box_zones, crs="EPSG:31370")
    gfo.to_file(
        gdf=gdf_zone, path=input_to_compare_with_path, create_spatial_index=True
    )

    input_to_select_from_path = tmp_path / "parcels.gpkg"
    box_parcels = []
    for parcel in parcels:
        box_parcel = shapely.from_wkt(parcel)
        box_parcels.append(box_parcel)
    gdf_parcel = gpd.GeoDataFrame(geometry=box_parcels, crs="EPSG:31370")
    gfo.to_file(
        gdf=gdf_parcel, path=input_to_select_from_path, create_spatial_index=True
    )

    # Test
    input_layerinfo = gfo.get_layerinfo(input_to_select_from_path)
    batchsize = math.ceil(input_layerinfo.featurecount / 2)
    output = "_".join(spatial_relations.split(" ")).lower()
    output_path = tmp_path / f"{output}.gpkg"
    gfo.export_by_location(
        input_to_select_from_path=str(input_to_select_from_path),
        input_to_compare_with_path=str(input_to_compare_with_path),
        output_path=str(output_path),
        spatial_relations_query=spatial_relations,
        batchsize=batchsize,
        subdivide_coords=subdivide_coords,
    )

    # Check if the output file is correctly created
    assert output_path.exists()
    output_layerinfo = gfo.get_layerinfo(output_path)
    assert output_layerinfo.featurecount == exp_features

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None
