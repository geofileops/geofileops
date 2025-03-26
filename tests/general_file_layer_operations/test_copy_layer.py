"""
Tests for functionalities in geofileops.general.
"""

import os
import shutil
from itertools import product
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry as sh_geom
from osgeo import gdal
from pandas.testing import assert_frame_equal
from pygeoops import GeometryType

import geofileops as gfo
from geofileops import fileops
from geofileops.util import _geofileinfo, _geopath_util, _geoseries_util
from tests import test_helper
from tests.test_helper import (
    SUFFIXES_FILEOPS,
    SUFFIXES_FILEOPS_EXT,
    assert_geodataframe_equal,
)

try:
    import fiona  # noqa: F401

    ENGINES = ["fiona", "pyogrio"]
except ImportError:
    ENGINES = ["pyogrio"]

gdal.UseExceptions()


@pytest.fixture(scope="module", params=ENGINES)
def engine_setter(request):
    engine = request.param
    engine_backup = os.environ.get("GFO_IO_ENGINE", None)
    if engine is None:
        del os.environ["GFO_IO_ENGINE"]
    else:
        os.environ["GFO_IO_ENGINE"] = engine
    yield engine
    if engine_backup is None:
        del os.environ["GFO_IO_ENGINE"]
    else:
        os.environ["GFO_IO_ENGINE"] = engine_backup


@pytest.fixture
def points_gdf():
    nb_points = 10
    gdf = gpd.GeoDataFrame(
        [
            {"geometry": sh_geom.Point(x, y), "value1": x + y, "value2": x * y}
            for x, y in zip(range(nb_points), range(nb_points))
        ],
        crs="epsg:4326",
    )
    return gdf


def test_append_to(tmp_path):
    """Test the append_to function.

    It is deprecated, but is still kept alive for backwards compatibility.
    """
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    dst = tmp_path / "output.gpkg"
    gfo.copy(src, dst)
    layer = gfo.get_only_layer(dst)

    # Append to file
    with pytest.warns(FutureWarning):
        gfo.append_to(src, dst, dst_layer=layer)

    # Test if number of rows is correct
    info = gfo.get_layerinfo(dst)
    assert info.featurecount == 96


def test_convert(tmp_path):
    """Test the convert function.

    The function is deprecated, but is still kept alive for backwards compatibility.
    """
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel")
    dst = tmp_path / "output.gpkg"

    # Test
    with pytest.warns(FutureWarning):
        gfo.convert(src, dst)

    # Now compare source and dst file
    src_layerinfo = gfo.get_layerinfo(src)
    dst_layerinfo = gfo.get_layerinfo(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert len(src_layerinfo.columns) == len(dst_layerinfo.columns)
    assert src_layerinfo.geometrytypename == dst_layerinfo.geometrytypename


@pytest.mark.parametrize(
    "testfile, suffix_input, suffix_output",
    [
        *product(["polygon-parcel"], SUFFIXES_FILEOPS_EXT, SUFFIXES_FILEOPS_EXT),
        ["curvepolygon", ".gpkg", ".gpkg"],
    ],
)
def test_copy_layer(tmp_path, testfile, suffix_input, suffix_output):
    if suffix_input == ".shp.zip" and suffix_output == ".shp":
        # GDAL < 3.10 determines the layer name wrong for .shp.zip leading to this error
        pytest.xfail("Copy of .shp.zip gives issues in GDAL <= 3.10")

    # Prepare test data
    src = test_helper.get_testfile(testfile, suffix=suffix_input)
    if suffix_input == ".csv" or suffix_output == ".csv":
        raise_on_nogeom = False
    else:
        raise_on_nogeom = True

    if suffix_input == ".csv" and suffix_output in (".shp", ".shp.zip"):
        # If no geometry column, there will only be a .dbf output file
        dst = tmp_path / f"{src.stem}-output.dbf"
    else:
        dst = tmp_path / f"{src.stem}-output{suffix_output}"

    # Test
    gfo.copy_layer(str(src), str(dst))

    # Now compare source and dst file
    src_layerinfo = gfo.get_layerinfo(src, raise_on_nogeom=raise_on_nogeom)
    dst_layerinfo = gfo.get_layerinfo(dst, raise_on_nogeom=raise_on_nogeom)
    assert dst_layerinfo.name == gfo.get_default_layer(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert len(src_layerinfo.columns) == len(dst_layerinfo.columns)
    if not (
        (suffix_input != ".csv" and suffix_output == ".csv")
        or (suffix_input == ".shp" and suffix_output == ".gpkg")
    ):
        assert src_layerinfo.geometrytypename == dst_layerinfo.geometrytypename


@pytest.mark.parametrize(
    "testfile, suffix_input, suffix_output",
    [
        *product(["polygon-parcel"], [".gpkg", ".shp"], [".gpkg", ".shp"]),
        ["curvepolygon", ".gpkg", ".gpkg"],
    ],
)
def test_copy_layer_dimensions_xyz(tmp_path, testfile, suffix_input, suffix_output):
    # Prepare test data
    src = test_helper.get_testfile(testfile, suffix=suffix_input, dimensions="XYZ")
    dst = tmp_path / f"{src.stem}-output{suffix_output}"

    # Test
    gfo.copy_layer(str(src), str(dst))

    # Now compare source and dst file
    src_layerinfo = gfo.get_layerinfo(src)
    dst_layerinfo = gfo.get_layerinfo(dst)
    assert dst_layerinfo.name == dst.stem
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert len(src_layerinfo.columns) == len(dst_layerinfo.columns)
    if not (suffix_input == ".shp" and suffix_output == ".gpkg"):
        assert src_layerinfo.geometrytypename == dst_layerinfo.geometrytypename


def test_copy_layer_add_layer_gpkg(tmp_path):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel")
    dst = tmp_path / "output.gpkg"
    layer1 = gfo.get_default_layer(dst)

    # First "add_layer" to file already while the file doesn't exist yet
    gfo.copy_layer(src, dst, dst_layer=layer1, write_mode="add_layer")

    # Check result
    layer1_info = gfo.get_layerinfo(dst, layer1)
    assert layer1_info.featurecount == 48

    # Now "add_layer" with the layer existing, and use force=True to overwrite it.
    # Use a filter so it is clear it was overwritten.
    gfo.copy_layer(
        src, dst, dst_layer=layer1, write_mode="add_layer", where="OIDN=1", force=True
    )

    # Check if the layer was properly overwritten
    assert gfo.listlayers(dst) == [layer1]
    layer1_info = gfo.get_layerinfo(dst, layer1)
    assert layer1_info.featurecount == 1

    # Now "add_layer" with the layer existing, and use force=False not to overwrite it.
    gfo.copy_layer(src, dst, dst_layer=layer1, write_mode="add_layer", force=False)

    # Check if the layer was properly overwritten
    assert gfo.listlayers(dst) == [layer1]
    layer1_info = gfo.get_layerinfo(dst, layer1)
    assert layer1_info.featurecount == 1

    # Finally "add_layer" to a new layer name
    layer2 = "new_layer"
    gfo.copy_layer(src, dst, dst_layer=layer2, write_mode="add_layer")

    # Check properties of both layers
    layer1_info = gfo.get_layerinfo(dst, layer1)
    assert layer1_info.featurecount == 1

    layer2_info = gfo.get_layerinfo(dst, layer2)
    assert layer2_info.featurecount == 48


def test_copy_layer_add_layer_shp(tmp_path):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel")
    dst = tmp_path / "output.shp"
    layer1 = gfo.get_default_layer(dst)

    # First "add_layer" to file already while the file doesn't exist yet
    gfo.copy_layer(src, dst, dst_layer=layer1, write_mode="add_layer")

    # Check result
    layer1_info = gfo.get_layerinfo(dst, layer1)
    assert layer1_info.featurecount == 48


def test_copy_layer_append_different_layer(tmp_path):
    # Prepare test data
    src_path = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    dst_path = tmp_path / "dst.gpkg"

    # Copy src file to dst file to "layer1"
    gfo.copy_layer(
        str(src_path), str(dst_path), dst_layer="layer1", write_mode="append"
    )
    src_info = gfo.get_layerinfo(src_path)
    dst_layer1_info = gfo.get_layerinfo(dst_path, "layer1")
    assert src_info.featurecount == dst_layer1_info.featurecount

    # Append src file layer to dst file to new layer: "layer2"
    gfo.copy_layer(src_path, dst_path, dst_layer="layer2", write_mode="append")
    dst_layer2_info = gfo.get_layerinfo(dst_path, "layer2")
    assert dst_layer1_info.featurecount == dst_layer2_info.featurecount


@pytest.mark.parametrize("suffix", SUFFIXES_FILEOPS)
def test_copy_layer_append_columns(tmp_path, suffix):
    """Test appending rows specifying some columns.

    This does not seem to be supported by GDAL.
    """
    # Prepare test data
    src_path = test_helper.get_testfile(
        "polygon-parcel", dst_dir=tmp_path, suffix=suffix
    )
    dst_path = tmp_path / f"dst{suffix}"
    gfo.copy(src_path, dst_path)

    src_info = gfo.get_layerinfo(src_path, raise_on_nogeom=False)
    src_columns = list(src_info.columns)
    dst_columns = ["OIDN", "UIDN", "GEWASGROEP"]
    for column in src_columns:
        if column not in dst_columns:
            gfo.drop_column(dst_path, column_name=column)

    # For GPKG and CSV files, the append fails
    if suffix in (".gpkg", ".csv"):
        pytest.xfail(
            "Appending only certain columns is not supported for GPKG and CSV files"
        )

    # For other file types, all rows are appended tot the dst layer, but the extra
    # column is not!
    gfo.copy_layer(src_path, dst_path, columns=dst_columns, write_mode="append")

    # Check results
    dst_info = gfo.get_layerinfo(dst_path, raise_on_nogeom=False)
    assert (src_info.featurecount * 2) == dst_info.featurecount
    assert len(dst_info.columns) == len(dst_columns)


@pytest.mark.parametrize("suffix", SUFFIXES_FILEOPS)
def test_copy_layer_append_default_layer(tmp_path, suffix):
    """Test appending rows to a file without specifying a layer name."""
    # Prepare test data
    src_path = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    dst_path = tmp_path / f"dst{suffix}"

    # Copy src file to dst file to "layer1"
    gfo.copy_layer(src_path, dst_path, write_mode="append")
    src_info = gfo.get_layerinfo(src_path, raise_on_nogeom=False)
    dst_info = gfo.get_layerinfo(dst_path, raise_on_nogeom=False)
    assert dst_info.featurecount == src_info.featurecount

    # Append src file layer to dst file to new layer: "layer2"
    gfo.copy_layer(src_path, dst_path, write_mode="append")
    dst_info = gfo.get_layerinfo(dst_path, raise_on_nogeom=False)
    assert dst_info.featurecount == src_info.featurecount * 2


@pytest.mark.parametrize("suffix", SUFFIXES_FILEOPS)
def test_copy_layer_append_different_columns(tmp_path, suffix):
    """Test appending rows to a file with a column less than in source file."""
    # Prepare test data
    src_path = test_helper.get_testfile(
        "polygon-parcel", dst_dir=tmp_path, suffix=suffix
    )
    dst_path = tmp_path / f"dst{suffix}"
    gfo.copy_layer(src_path, dst_path)
    gfo.add_column(src_path, name="extra_col", type=gfo.DataType.INTEGER)

    # All rows are appended tot the dst layer, but the extra column is not!
    gfo.copy_layer(src_path, dst_path, write_mode="append")

    # Check results
    raise_on_nogeom = False if suffix == ".csv" else True

    src_info = gfo.get_layerinfo(src_path, raise_on_nogeom=raise_on_nogeom)
    res_info = gfo.get_layerinfo(dst_path, raise_on_nogeom=raise_on_nogeom)
    assert (src_info.featurecount * 2) == res_info.featurecount
    assert len(src_info.columns) == len(res_info.columns) + 1


def test_copy_layer_append_error_non_default_layer(tmp_path):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    dst = tmp_path / "output.gpkg"
    gfo.copy(src, dst)

    # Append fails if no layer is specified and a layer that does not have the default
    # layer name exists already
    with pytest.raises(ValueError, match="dst_layer is required when write_mode is"):
        gfo.copy_layer(src, dst, write_mode="append")


def test_copy_layer_append_error_other_layers(tmp_path):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    dst = tmp_path / "output.gpkg"
    gfo.copy(src, dst)
    gfo.copy_layer(
        src, dst, write_mode="add_layer", dst_layer=gfo.get_default_layer(dst)
    )

    # Append fails if no layer is specified and multiple layers exist already, even if
    # one of them has the default layer name
    with pytest.raises(ValueError, match="dst_layer is required when write_mode is"):
        gfo.copy_layer(src, dst, write_mode="append")


@pytest.mark.parametrize("testfile", ["polygon-parcel", "curvepolygon"])
def test_copy_layer_append_shp_laundered_columns(tmp_path, testfile):
    # GDAL doesn't seem to handle appending to a shapefile where column laundering is
    # needed very well: all laundered columns get NULL values instead of the actual
    # values.
    # gfo.append_to bypasses this by laundering the columns beforehand via an sql
    # statement so gdal doesn't need to do laundering.
    # Start from a gpkg test file, because that can have long column names that need
    # laundering.
    src_path = test_helper.get_testfile(testfile, dst_dir=tmp_path, suffix=".gpkg")
    gfo.add_column(
        src_path, name="extra_long_columnname", type="TEXT", expression="'TEST VALUE'"
    )
    dst_path = tmp_path / "dst.shp"
    gfo.copy_layer(src_path, dst_path, write_mode="append")
    gfo.copy_layer(src_path, dst_path, write_mode="append")

    src_info = gfo.get_layerinfo(src_path)
    dst_info = gfo.get_layerinfo(dst_path)

    # All rows are appended tot the dst layer, and long column name should be laundered
    assert (src_info.featurecount * 2) == dst_info.featurecount
    assert len(src_info.columns) == len(dst_info.columns)
    assert "extra_long" in dst_info.columns
    assert "extra_long_columnname" not in dst_info.columns

    # Check content
    dst_gdf = gfo.read_file(dst_path)
    assert dst_gdf is not None
    assert dst_gdf["extra_long"].to_list() == ["TEST VALUE"] * len(dst_gdf)


@pytest.mark.parametrize("create_spatial_index", [True, False])
def test_copy_layer_append_spatial_index(tmp_path, create_spatial_index):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel")
    dst = tmp_path / "output.gpkg"
    layer1 = gfo.get_default_layer(dst)

    if create_spatial_index is not None:
        index_expected = create_spatial_index
    else:
        index_expected = _geofileinfo.get_geofileinfo(dst).default_spatial_index

    # First append to file already while it doesn't exist yet
    gfo.copy_layer(
        src, dst, write_mode="append", create_spatial_index=create_spatial_index
    )

    # Check result
    layer1_info = gfo.get_layerinfo(dst)
    assert layer1_info.featurecount == 48
    assert gfo.has_spatial_index(dst) == index_expected

    # Now append while the file exists to the existing layer
    gfo.copy_layer(
        src, dst, write_mode="append", create_spatial_index=create_spatial_index
    )

    # Check if number of rows is correct
    layer1_info = gfo.get_layerinfo(dst)
    assert layer1_info.featurecount == 96
    assert gfo.has_spatial_index(dst) == index_expected

    # Finally append while the file exists but to a new layer
    layer2 = "new_layer"
    gfo.copy_layer(
        src,
        dst,
        dst_layer=layer2,
        write_mode="append",
        create_spatial_index=create_spatial_index,
    )

    # Check properties of both layers
    layer1_info = gfo.get_layerinfo(dst, layer1)
    assert layer1_info.featurecount == 96
    assert gfo.has_spatial_index(dst, layer1) == index_expected

    layer2_info = gfo.get_layerinfo(dst, layer2)
    assert layer2_info.featurecount == 48
    assert gfo.has_spatial_index(dst, layer2) == index_expected


@pytest.mark.parametrize("suffix", [s for s in SUFFIXES_FILEOPS if s != ".csv"])
@pytest.mark.parametrize("columns", [["OIDN", "UIDN"], "OIDN"])
def test_copy_layer_columns(tmp_path, suffix, columns):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel", suffix=suffix)

    # Test
    dst = tmp_path / f"output{suffix}"
    gfo.copy_layer(src, dst, columns=columns)
    copy_gdf = gfo.read_file(dst)
    input_gdf = gfo.read_file(src, columns=columns)
    assert_geodataframe_equal(input_gdf, copy_gdf)


@pytest.mark.parametrize("suffix", SUFFIXES_FILEOPS)
@pytest.mark.parametrize("dimensions", [None, "XYZ"])
def test_copy_layer_emptyfile(tmp_path, dimensions, suffix):
    # Prepare test data
    src = test_helper.get_testfile(
        "polygon-parcel",
        suffix=suffix,
        dst_dir=tmp_path,
        empty=True,
        dimensions=dimensions,
    )
    raise_on_nogeom = False if suffix == ".csv" else True
    dst = tmp_path / f"{src.stem}-output{suffix}"

    # Convert
    gfo.copy_layer(src, dst)

    # Now compare source and dst file
    assert dst.exists()
    src_layerinfo = gfo.get_layerinfo(src, raise_on_nogeom=raise_on_nogeom)
    dst_layerinfo = gfo.get_layerinfo(dst, raise_on_nogeom=raise_on_nogeom)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert len(src_layerinfo.columns) == len(dst_layerinfo.columns)
    assert src_layerinfo.geometrytypename == dst_layerinfo.geometrytypename


@pytest.mark.parametrize(
    "testfile, expected_count", [("polygon-parcel", 50), ("point", 50)]
)
def test_copy_layer_explodecollections(tmp_path, testfile, expected_count):
    # Prepare test data
    src = test_helper.get_testfile(testfile)
    dst = tmp_path / f"{src.stem}.gpkg"

    # copy_layer, with explodecollections. Default behaviour of gdal was to try to
    # preserve the fids, but this didn't work with explodecolledtions, this was
    # overruled in #395
    gfo.copy_layer(src, dst, explodecollections=True)

    result_gdf = gfo.read_file(dst)
    assert len(result_gdf) == expected_count


@pytest.mark.parametrize(
    "testfile, force_geometrytype",
    [
        ("polygon-parcel", GeometryType.POLYGON),
        ("polygon-parcel", GeometryType.MULTIPOLYGON),
        ("polygon-parcel", GeometryType.MULTILINESTRING),
        ("polygon-parcel", GeometryType.MULTIPOINT),
    ],
)
@pytest.mark.filterwarnings(
    "ignore:.*A geometry of type MULTIPOLYGON is inserted into .*"
)
def test_copy_layer_force_output_geometrytype(tmp_path, testfile, force_geometrytype):
    # The conversion is done by ogr, and the "test" is rather written to
    # explore the behaviour of this ogr functionality:
    # Single-part polygons are converted to the destination types, but multipolygons
    # are kept as they are.
    # Issue opened for this: https://github.com/OSGeo/gdal/issues/11068

    # copy_layer on testfile and force to force_geometrytype
    src = test_helper.get_testfile(testfile)
    dst = tmp_path / f"{src.stem}_to_{force_geometrytype}.gpkg"
    gfo.copy_layer(src, dst, force_output_geometrytype=force_geometrytype)
    assert gfo.get_layerinfo(dst).geometrytype == force_geometrytype

    result_gdf = gfo.read_file(dst)
    assert len(result_gdf) == 48


@pytest.mark.parametrize(
    "kwargs, exp_ex, exp_error",
    [
        ({"src": "non_existing_file.gpkg"}, FileNotFoundError, "File not found"),
        ({"write_mode": "invalid"}, ValueError, "Invalid write_mode"),
        (
            {"write_mode": "add_layer"},
            ValueError,
            "dst_layer is required when write_mode is",
        ),
        (
            {"write_mode": "append", "append": True},
            ValueError,
            "append parameter is deprecated, use write_mode",
        ),
    ],
)
def test_copy_layer_errors(tmp_path, kwargs, exp_ex, exp_error):
    # Convert
    if "src" not in kwargs:
        kwargs["src"] = test_helper.get_testfile("polygon-parcel")
    kwargs["dst"] = tmp_path / "output.gpkg"

    with pytest.raises(exp_ex, match=exp_error):
        gfo.copy_layer(**kwargs)


def test_copy_layer_input_open_options(tmp_path):
    # Prepare test data
    src = tmp_path / "input.csv"
    dst = tmp_path / "output.gpkg"
    with open(src, "w") as srcfile:
        srcfile.write("POINT_ID, POINT_LAT, POINT_LON, POINT_NAME\n")
        srcfile.write('1, 50.939972761,3.888498686, "random spot"\n')

    # copy_layer with open_options
    gfo.copy_layer(
        src,
        dst,
        options={
            "INPUT_OPEN.X_POSSIBLE_NAMES": "POINT_LON",
            "INPUT_OPEN.Y_POSSIBLE_NAMES": "POINT_LAT",
        },
        dst_crs="EPSG:4326",
    )

    # Check result
    assert dst.exists()
    result_gdf = gfo.read_file(dst)
    assert len(result_gdf) == 1
    assert "geometry" in result_gdf.columns
    assert result_gdf.geometry[0].x == 3.888498686
    assert result_gdf.geometry[0].y == 50.939972761


@pytest.mark.parametrize("layer", [None, "parcels_output"])
def test_copy_layer_layer(tmp_path, layer):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel")
    dst = tmp_path / "output.gpkg"
    expected_layer = dst.stem if layer is None else layer

    gfo.copy_layer(src, dst, dst_layer=layer)

    # Check result
    dst_info = gfo.get_layerinfo(dst)
    assert dst_info.name == expected_layer
    assert dst_info.featurecount == 48


@pytest.mark.parametrize(
    "src_suffix, dst_suffix, preserve_fid, exp_preserved_fids",
    [
        (".shp", ".gpkg", True, True),
        (".shp", ".gpkg", False, False),
        (".gpkg", ".shp", True, False),
        (".shp", ".sqlite", True, True),
        (".shp", ".sqlite", False, False),
    ],
)
def test_copy_layer_preserve_fid(
    tmp_path,
    src_suffix: str,
    dst_suffix: str,
    preserve_fid: bool,
    exp_preserved_fids: bool,
):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel", suffix=src_suffix)
    dst = tmp_path / f"{src.stem}-output_preserve_fid-{preserve_fid}{dst_suffix}"

    # copy_layer
    gfo.copy_layer(src, dst, preserve_fid=preserve_fid)

    # Now compare source and dst file
    src_gdf = gfo.read_file(src, fid_as_index=True)
    dst_gdf = gfo.read_file(dst, fid_as_index=True)
    assert len(src_gdf) == len(dst_gdf)
    assert len(src_gdf.columns) == len(dst_gdf.columns)
    if exp_preserved_fids:
        assert src_gdf.index.tolist() == dst_gdf.index.tolist()
    else:
        assert src_gdf.index.tolist() != dst_gdf.index.tolist()


@pytest.mark.parametrize("suffix", [s for s in SUFFIXES_FILEOPS if s != ".csv"])
@pytest.mark.parametrize("src_crs", [None, 31370])
def test_copy_layer_reproject(tmp_path, suffix, src_crs):
    src = test_helper.get_testfile("polygon-parcel", suffix=suffix)
    dst = tmp_path / f"{src.stem}-output_reproj4326{suffix}"

    # copy_layer with reproject
    gfo.copy_layer(src, dst, src_crs=src_crs, dst_crs=4326, reproject=True)

    # Now compare source and dst file
    src_layerinfo = gfo.get_layerinfo(src)
    dst_layerinfo = gfo.get_layerinfo(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount
    assert src_layerinfo.crs is not None
    assert src_layerinfo.crs.to_epsg() == 31370
    assert dst_layerinfo.crs is not None
    assert dst_layerinfo.crs.to_epsg() == 4326

    # Check if dst file actually seems to contain lat lon coordinates
    dst_gdf = gfo.read_file(dst)
    first_geom = dst_gdf.geometry[0]
    first_poly = (
        first_geom if isinstance(first_geom, sh_geom.Polygon) else first_geom.geoms[0]
    )
    assert first_poly.exterior is not None
    for x, y in first_poly.exterior.coords:
        assert x < 100 and y < 100


@pytest.mark.parametrize("suffix", SUFFIXES_FILEOPS)
def test_copy_layer_sql(tmp_path, suffix):
    # Prepare test data
    # For multi-layer filetype, use 2-layer file for better test coverage
    if _geofileinfo.get_geofileinfo(suffix).is_singlelayer:
        testfile = "polygon-parcel"
        src = test_helper.get_testfile(testfile, suffix=suffix)
        src_layer = src.stem
    else:
        testfile = "polygon-twolayers"
        src = test_helper.get_testfile(testfile, suffix=suffix)
        src_layer = "parcels"

    # Test
    sql_stmt = f'SELECT * FROM "{src_layer}"'
    dst = tmp_path / f"output{suffix}"
    gfo.copy_layer(src, dst, src_layer=src_layer, sql_stmt=sql_stmt)
    read_gdf = gfo.read_file(src, sql_stmt=sql_stmt)
    assert isinstance(read_gdf, pd.DataFrame)
    if not suffix == ".csv":
        assert isinstance(read_gdf, gpd.GeoDataFrame)
    assert len(read_gdf) == 48


@pytest.mark.parametrize("suffix", [s for s in SUFFIXES_FILEOPS if s != ".csv"])
def test_copy_layer_sql_placeholders(tmp_path, suffix):
    """
    Test if placeholders are properly filled out + if casing used in columns parameter
    is retained when using placeholders.
    """
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel", suffix=suffix)

    # Test
    sql_stmt = """
        SELECT {geometrycolumn} AS geom
              {columns_to_select_str}
          FROM "{input_layer}"
    """
    dst = tmp_path / f"output{suffix}"
    gfo.copy_layer(src, dst, sql_stmt=sql_stmt, sql_dialect="SQLITE", columns=["OIDN"])
    copy_gdf = gfo.read_file(dst)
    input_gdf = gfo.read_file(src, columns=["OIDN"])
    assert_geodataframe_equal(input_gdf, copy_gdf)


def test_copy_layer_to_gpkg_zip(tmp_path):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel")
    dst = tmp_path / "output.gpkg.zip"

    # copy_layer
    gfo.copy_layer(src, dst)

    # Now compare source and dst file
    src_layerinfo = gfo.get_layerinfo(src)
    dst_layerinfo = gfo.get_layerinfo(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount

    src_gdf = gfo.read_file(src)
    dst_gdf = gfo.read_file(dst)
    assert_geodataframe_equal(src_gdf, dst_gdf)


def test_copy_layer_vsi(tmp_path):
    # Prepare test data
    src = f"/vsizip//vsicurl/{test_helper.data_url}/poly_shp.zip/poly.shp"
    dst = tmp_path / "output.gpkg"

    # copy_layer with vsi
    gfo.copy_layer(src, dst)

    # Now compare source and dst file
    src_layerinfo = gfo.get_layerinfo(src)
    dst_layerinfo = gfo.get_layerinfo(dst)
    assert src_layerinfo.featurecount == dst_layerinfo.featurecount


@pytest.mark.parametrize("suffix", SUFFIXES_FILEOPS)
def test_copy_layer_where(tmp_path, suffix):
    # Prepare test data
    src = test_helper.get_testfile("polygon-parcel", suffix=suffix)
    if suffix == ".csv":
        where = "GEWASGROEP = 'Grasland'"
        exp_featurecount = 31
        raise_on_nogeom = False
    else:
        where = "ST_Area({geometrycolumn}) > 500"
        exp_featurecount = 43
        raise_on_nogeom = True

    # copy_layer with where
    dst = tmp_path / f"{src.stem}-output_where{suffix}"
    gfo.copy_layer(src, dst, where=where)

    # Now compare source and dst file
    src_layerinfo = gfo.get_layerinfo(src, raise_on_nogeom=raise_on_nogeom)
    dst_layerinfo = gfo.get_layerinfo(dst, raise_on_nogeom=raise_on_nogeom)
    assert src_layerinfo.featurecount > dst_layerinfo.featurecount
    assert dst_layerinfo.featurecount == exp_featurecount


def test_copy_layer_write_mode_add_layer(tmp_path):
    src = test_helper.get_testfile("polygon-parcel")
    dst = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    dst_layer = "parcels_2"

    # Test
    gfo.copy_layer(src, dst, dst_layer=dst_layer, write_mode="add_layer")

    # Test if number of rows is correct
    layers = gfo.listlayers(dst)
    assert len(layers) == 2
    assert dst_layer in layers

    info = gfo.get_layerinfo(dst, layer=dst_layer)
    assert info.featurecount == 48


def test_copy_layer_write_mode_append(tmp_path):
    src = test_helper.get_testfile("polygon-parcel")
    dst = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    dst_layer = gfo.get_only_layer(dst)

    # Test
    gfo.copy_layer(src, dst, dst_layer=dst_layer, write_mode="append")

    # Test if number of rows is correct
    info = gfo.get_layerinfo(dst)
    assert info.featurecount == 96


@pytest.mark.parametrize("write_mode", [None, "create", "add_layer"])
@pytest.mark.parametrize("force", [True, False])
def test_copy_layer_write_mode_force(tmp_path, write_mode, force):
    """Test if force parameter is properly handled."""
    src = test_helper.get_testfile("polygon-parcel")
    dst = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    kwargs = {}
    if write_mode is not None:
        kwargs["write_mode"] = write_mode
    dst_layer = gfo.get_only_layer(dst)

    mtime_orig = dst.stat().st_mtime
    gfo.copy_layer(src, dst, dst_layer=dst_layer, force=force, **kwargs)
    if force:
        assert dst.stat().st_mtime > mtime_orig
    else:
        assert dst.stat().st_mtime == mtime_orig
