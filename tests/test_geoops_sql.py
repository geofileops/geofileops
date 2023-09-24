import pytest

from geofileops.util import _geoops_sql


@pytest.mark.parametrize(
    "descr, nb_rows_input_layer, nb_parallel, batchsize, is_twolayer_operation, "
    "exp_nb_parallel, exp_nb_batches",
    [
        ("0 input rows, batchsize=1, singlelayer", 0, 2, 1, 0, 1, 1),
        ("0 input rows, nb_parallel=2, singlelayer", 0, 2, 0, 0, 1, 1),
        ("0 input rows, singlelayer", 0, -1, -1, 0, 1, 1),
        ("0 input rows, twolayer", 0, -1, -1, 1, 1, 1),
        ("1 input row, batchsize=1, singlelayer", 1, 2, 1, 0, 1, 1),
        ("1 input rows, nb_parallel=2, singlelayer", 1, 2, 0, 0, 1, 1),
        ("1 input rows, singlelayer", 1, -1, -1, 0, 1, 1),
        ("1 input rows, twolayer", 1, -1, -1, 1, 1, 1),
    ],
)
def test_determine_nb_batches(
    descr: str,
    nb_rows_input_layer: int,
    nb_parallel: int,
    batchsize: int,
    is_twolayer_operation: bool,
    exp_nb_parallel: int,
    exp_nb_batches: int,
):
    res_nb_parallel, res_nb_batches = _geoops_sql._determine_nb_batches(
        nb_rows_input_layer=nb_rows_input_layer,
        nb_parallel=nb_parallel,
        batchsize=batchsize,
        is_twolayer_operation=is_twolayer_operation,
    )
    assert exp_nb_parallel == res_nb_parallel
    assert exp_nb_batches == res_nb_batches
