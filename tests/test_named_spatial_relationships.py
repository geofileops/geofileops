import math

import geopandas as gpd
import pytest
import shapely

import geofileops as gfo

zones = [
    "POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0))",
    "POLYGON ((10 1, 13 1, 13 4, 10 4, 10 1))",
]

extra_parcels = [
    "POLYGON ((-1 5, 1 5, 1 7, -1 7, -1 5))",
    "POLYGON ((3 6, 5 6, 5 8, 3 8, 3 6))",
]


@pytest.mark.parametrize("zones", [zones])
@pytest.mark.parametrize(
    "parcels", [["POLYGON ((2 1, 4 1, 4 3, 2 3, 2 1))", *extra_parcels]]
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("intersects is True", 1), ("intersects is False", 2)],
)
def test_intertsects(
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
            "POLYGON ((9 0, 11 0, 11 2, 9 2, 9 0))",
            *extra_parcels,
        ]
    ],
)
@pytest.mark.parametrize("subdivide_coords", [0, 10])
@pytest.mark.parametrize(
    "spatial_relations, exp_features",
    [("disjoint is True", 3), ("disjoint is False", 1)],
)
def test_disjoint(
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
    gfo.to_file(gdf=gdf_zone, path=input_to_compare_with_path)

    input_to_select_from_path = tmp_path / "parcels.gpkg"
    box_parcels = []
    for parcel in parcels:
        box_parcel = shapely.from_wkt(parcel)
        box_parcels.append(box_parcel)
    gdf_parcel = gpd.GeoDataFrame(geometry=box_parcels, crs="EPSG:31370")
    gfo.to_file(gdf=gdf_parcel, path=input_to_select_from_path)

    # DEBUG:
    gfo.to_file(
        gdf=gdf_zone,
        path=tmp_path / "output.gpkg",
        layer="output",
    )
    gfo.to_file(
        gdf=gdf_parcel,
        path=tmp_path / "output.gpkg",
        layer="output",
        append=True,
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

    # Check if the output file is c1orrectly created
    assert output_path.exists()
    output_layerinfo = gfo.get_layerinfo(output_path)
    assert output_layerinfo.featurecount == exp_features

    # Check the contents of the result file
    output_gdf = gfo.read_file(output_path)
    assert output_gdf["geometry"][0] is not None
