import pytest

from geofileops.util import _geoops_gpd


@pytest.mark.parametrize(
    "descr, nb_rows_input_layer, nb_parallel, batchsize, bytes_usable, "
    "exp_nb_parallel, exp_nb_batches",
    [
        ("0 input rows, batchsize=1", 0, 2, 1, 500_000_000, 1, 1),
        ("0 input rows, nb_parallel=2", 0, 2, 0, 500_000_000, 1, 1),
        ("0 input rows", 0, -1, -1, 500_000_000, 1, 1),
        ("1 input row, batchsize=1", 1, 2, 1, 500_000_000, 1, 1),
        ("1 input rows, nb_parallel=2", 1, 2, 0, 500_000_000, 1, 1),
        ("1 input rows", 1, -1, -1, 500_000_000, 1, 1),
        ("1 input rows", 1, -1, -1, 500_000_000, 1, 1),
        ("2 input rows, nb_parallel=10", 2, 10, -1, 500_000_000, 2, 2),
        ("100 input row, batchsize=20", 100, -1, 20, 500_000_000, 5, 5),
        ("100 input rows, nb_parallel=2", 100, 2, 0, 500_000_000, 2, 2),
        ("100 input rows, nb_parallel=1", 100, 1, 0, 500_000_000, 1, 1),
        ("100 input rows", 100, -1, -1, 500_000_000, 1, 1),
        ("1000 input row, batchsize=20", 1000, -1, 20, 500_000_000, 8, 56),
        ("1000 input rows, nb_parallel=2", 1000, 2, 0, 500_000_000, 2, 2),
        ("1000 input rows, nb_parallel=1", 1000, 1, 0, 500_000_000, 1, 1),
        ("1000 input rows", 1000, -1, -1, 500_000_000, 1, 1),
        ("100_000 input rows", 100000, -1, -1, 500_000_000, 8, 16),
        ("100_000 input rows, 250MB usable RAM", 100000, -1, -1, 250_000_000, 4, 12),
    ],
)
def test_determine_nb_batches(
    descr: str,
    nb_rows_input_layer: int,
    nb_parallel: int,
    batchsize: int,
    bytes_usable: float,
    exp_nb_parallel: int,
    exp_nb_batches: int,
):
    config = _geoops_gpd.ParallelizationConfig(bytes_usable=bytes_usable)
    assert config.bytes_usable == bytes_usable

    res_nb_parallel, res_nb_batches = _geoops_gpd._determine_nb_batches(
        nb_rows_total=nb_rows_input_layer,
        nb_parallel=nb_parallel,
        batchsize=batchsize,
        parallelization_config=config,
    )
    assert exp_nb_parallel == res_nb_parallel
    assert exp_nb_batches == res_nb_batches
