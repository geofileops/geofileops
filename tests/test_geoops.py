import pytest

import geofileops as gfo
from tests import test_helper


@pytest.mark.parametrize(
    "testfile, input_layer, output_layer, close_input_boundary_gaps, exp_featurecount",
    [
        ("polygon-twolayers", "parcels", "output_layername", True, 9),
        ("polygon-parcel", None, None, False, 11),
    ],
)
def test_dissolve_within_distance(
    tmp_path,
    testfile,
    input_layer,
    close_input_boundary_gaps,
    output_layer,
    exp_featurecount,
):
    input_path = test_helper.get_testfile(testfile)
    output_path = tmp_path / "output.gpkg"
    gfo.dissolve_within_distance(
        input_path=input_path,
        input_layer=input_layer,
        output_path=output_path,
        distance=10,
        gridsize=0.0001,
        close_input_boundary_gaps=close_input_boundary_gaps,
        output_layer=output_layer,
    )

    # Check result
    assert output_path.exists()
    if output_layer is None:
        # If no layername specified, layer has the default lauer name
        assert gfo.listlayers(output_path) == [output_path.stem]
    else:
        assert gfo.listlayers(output_path) == [output_layer]
    output_info = gfo.get_layerinfo(output_path)
    assert output_info.featurecount == exp_featurecount
