"""Tests for the general_helper module."""

import pytest

from geofileops.helpers import _general_helper
from geofileops.util import _general_util


def test_create_gfo_tmp_dir(tmp_path):
    """Test the creation of a temporary directory in the default tempdir."""
    with _general_util.TempEnv({"GFO_TMPDIR": None}):
        tempdir = _general_helper.create_gfo_tmp_dir("testje")

    assert tempdir.exists()
    assert tempdir.parent.name == "geofileops"
    assert tempdir.name.startswith("testje")


def test_create_gfo_tmp_dir_env(tmp_path):
    """Test the creation of a temporary directory in a dir specified via GFO_TMPDIR."""
    with _general_util.TempEnv({"GFO_TMPDIR": str(tmp_path)}):
        tempdir = _general_helper.create_gfo_tmp_dir("testje")

    assert tempdir.exists()
    assert str(tempdir).startswith(str(tmp_path))
    assert tempdir.name.startswith("testje")


def test_create_gfo_tmp_dir_env_invalid(tmp_path):
    """Test the creation of a temporary directory if GFO_TMPDIR is invalid."""
    # GFO_TMPDIR set to an empty string is not supported.
    with _general_util.TempEnv({"GFO_TMPDIR": ""}):
        with pytest.raises(
            RuntimeError,
            match="GFO_TMPDIR='' environment variable found which is not supported",
        ):
            _general_helper.create_gfo_tmp_dir("testje")


@pytest.mark.parametrize(
    "worker_type, input_layer_featurecount, expected",
    [
        ("processes", 1, "processes"),
        ("threads", 101, "threads"),
        ("auto", 1, "threads"),
        ("auto", 100, "threads"),
        ("auto", 101, "processes"),
    ],
)
def test_worker_type_to_use(worker_type, input_layer_featurecount, expected):
    with _general_util.TempEnv({"GFO_WORKER_TYPE": worker_type}):
        assert _general_helper.worker_type_to_use(input_layer_featurecount) == expected
