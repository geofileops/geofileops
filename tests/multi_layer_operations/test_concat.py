"""Tests for the concat function."""

import pytest

import geofileops as gfo
from geofileops.util import _geofileinfo
from tests import test_helper


@pytest.mark.parametrize(
    "columns, spatial_index, input_layers, output_layer, output_suffix",
    [
        (None, None, [None, "parcels", "parcels"], None, ".gpkg"),
        (None, None, "parcels", None, ".shp"),
        (["OIDN", "UIDN"], False, "parcels", "custom", ".gpkg"),
        (["OIDN"], True, ["parcels", "parcels", "parcels"], None, ".shp"),
    ],
)
def test_concat(
    tmp_path, columns, spatial_index, input_layers, output_layer, output_suffix
):
    """Test the concat function."""
    # Prepare test data
    input1 = test_helper.get_testfile("polygon-parcel")
    input2 = test_helper.get_testfile("polygon-twolayers")
    output = tmp_path / f"output{output_suffix}"

    # Test
    gfo.concat(
        [input1, input1, input2],
        output,
        input_layers=input_layers,
        output_layer=output_layer,
        columns=columns,
        create_spatial_index=spatial_index,
    )

    # Now check result
    input1_layerinfo = gfo.get_layerinfo(input1)
    input2_layerinfo = gfo.get_layerinfo(input2, layer="parcels")
    output_layerinfo = gfo.get_layerinfo(output)

    exp_featurecount = input1_layerinfo.featurecount * 2 + input2_layerinfo.featurecount
    assert output_layerinfo.featurecount == exp_featurecount
    exp_nb_columns = len(input1_layerinfo.columns) if columns is None else len(columns)
    assert len(output_layerinfo.columns) == exp_nb_columns
    assert output_layerinfo.geometrytypename == input1_layerinfo.geometrytypename

    exp_layer = (
        output_layer if output_layer is not None else gfo.get_default_layer(output)
    )
    assert output_layerinfo.name == exp_layer

    # Check spatial index
    exp_spatial_index = (
        spatial_index
        if spatial_index is not None
        else _geofileinfo.get_geofileinfo(output).default_spatial_index
    )
    assert gfo.has_spatial_index(output) is exp_spatial_index


def test_concat_explodecollections(tmp_path):
    """Test the concat function with explodecollections."""
    # Prepare test data
    input = test_helper.get_testfile("polygon-parcel")
    input_expl = tmp_path / "input_exploded.gpkg"
    gfo.copy_layer(input, input_expl, explodecollections=True)

    # Test
    output = tmp_path / "output.gpkg"
    gfo.concat([input_expl, input_expl], output, explodecollections=True)

    # Now check the result file
    input_info = gfo.get_layerinfo(input)
    input_expl_info = gfo.get_layerinfo(input_expl)
    output_info = gfo.get_layerinfo(output)

    # Make sure the input file contains multipolygons
    assert input_info.geometrytypename == "MULTIPOLYGON"
    assert input_info.featurecount < input_expl_info.featurecount

    exp_featurecount = input_expl_info.featurecount * 2
    assert output_info.featurecount == exp_featurecount
    exp_nb_column = len(input_expl_info.columns)
    assert len(output_info.columns) == exp_nb_column
    assert output_info.geometrytypename == "POLYGON"


def test_concat_input1_less_columns(tmp_path):
    """Test how concat deals with different columns in input files."""
    # Prepare test data
    input1 = test_helper.get_testfile("polygon-parcel", dst_dir=tmp_path)
    input2 = test_helper.get_testfile("polygon-twolayers")
    gfo.drop_column(input1, column_name="OIDN", layer="parcels")

    # Test
    output = tmp_path / "output_input1_less_columns.gpkg"
    gfo.concat([input1, input2], output, input_layers=[None, "parcels"])

    # Now check result file
    input1_info = gfo.get_layerinfo(input1)
    input2_info = gfo.get_layerinfo(input2, layer="parcels")
    output_info = gfo.get_layerinfo(output)

    exp_featurecount = input1_info.featurecount + input2_info.featurecount
    assert output_info.featurecount == exp_featurecount
    exp_nb_column = len(input1_info.columns)
    assert len(output_info.columns) == exp_nb_column
    assert output_info.geometrytypename == input1_info.geometrytypename
    assert "OIDN" not in output_info.columns


def test_concat_input2_less_columns(tmp_path):
    """Test how concat deals with different columns in input files."""
    # Prepare test data
    input1 = test_helper.get_testfile("polygon-parcel")
    input2 = test_helper.get_testfile("polygon-twolayers", dst_dir=tmp_path)
    gfo.drop_column(input2, column_name="OIDN", layer="parcels")

    # Test
    output = tmp_path / "output_input2_less_columns.gpkg"
    gfo.concat([input1, input2], output, input_layers=[None, "parcels"])

    # Now check result file
    input1_info = gfo.get_layerinfo(input1)
    input2_info = gfo.get_layerinfo(input2, layer="parcels")
    output_info = gfo.get_layerinfo(output)

    exp_featurecount = input1_info.featurecount + input2_info.featurecount
    assert output_info.featurecount == exp_featurecount
    exp_nb_column = len(input1_info.columns)
    assert len(output_info.columns) == exp_nb_column
    assert output_info.geometrytypename == input1_info.geometrytypename

    # The missing column in input2 should be filled with null values
    output_gdf = gfo.read_file(output)
    assert output_gdf["OIDN"].isnull().sum() == input2_info.featurecount


def test_concat_invalid_input(tmp_path):
    """Test the concat function with invalid input."""
    # Prepare test data
    input = test_helper.get_testfile("polygon-parcel")

    # Test
    output = tmp_path / "output.gpkg"
    with pytest.raises(
        ValueError,
        match="input_layers must have the same length as input_paths if specified",
    ):
        gfo.concat([input, input], output, input_layers=["lines"])


def test_concat_output_exists(tmp_path):
    """Test the concat function with an existing output file."""
    # Prepare test data
    input = test_helper.get_testfile("polygon-parcel")
    output = tmp_path / "output.gpkg"
    output.touch()

    # Test
    gfo.concat([input, input], output)

    # The test file should not be overwritten/changed.
    assert output.stat().st_size == 0
