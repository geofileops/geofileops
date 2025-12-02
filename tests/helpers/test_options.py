"""Tests for functionalities in _configoptions_helper."""

import os
import tempfile

import pytest
from pyproj import CRS

import geofileops as gfo
from geofileops.helpers import _options
from geofileops.helpers._options import Options


@pytest.mark.parametrize(
    "value, default, expected",
    [
        ("TruE", False, True),
        ("YeS", False, True),
        ("1", False, True),
        ("nO", True, False),
        ("FalsE", True, False),
        ("0", True, False),
        ("", True, True),
        ("", False, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_get_bool(value, default, expected):
    test_key = "GFO_TEST_BOOL"
    if value is None:
        if test_key in os.environ:
            del os.environ[test_key]
    else:
        os.environ[test_key] = value

    result = _options._get_bool(test_key, default=default)
    if test_key in os.environ:
        del os.environ[test_key]

    assert result is expected


def test_get_bool_invalidvalue():
    test_key = "GFO_TEST_BOOL"
    with (
        gfo.TempEnv({test_key: "INVALID"}),
        pytest.raises(ValueError, match="invalid value for bool configoption"),
    ):
        _ = _options._get_bool(test_key, default="")


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("GFO_IO_ENGINE", "PYOgrio", "pyogrio"),
        ("GFO_IO_ENGINE", "FIOna", "fiona"),
        ("GFO_IO_ENGINE", None, "pyogrio-arrow"),
        ("GFO_ON_DATA_ERROR", "RAIse", "raise"),
        ("GFO_ON_DATA_ERROR", "WARn", "warn"),
        ("GFO_ON_DATA_ERROR", None, "raise"),
        ("GFO_REMOVE_TEMP_FILES", "TRUe", True),
        ("GFO_REMOVE_TEMP_FILES", "FALse", False),
        ("GFO_REMOVE_TEMP_FILES", None, True),
        ("GFO_WORKER_TYPE", "THReads", "threads"),
        ("GFO_WORKER_TYPE", "PROcesses", "processes"),
        ("GFO_WORKER_TYPE", "AUTo", "auto"),
        ("GFO_WORKER_TYPE", None, "auto"),
    ],
)
def test_configoptions(key, value, expected):
    """Test all ConfigOptions class properties."""
    with gfo.TempEnv({key: value}):
        if key == "GFO_IO_ENGINE":
            result = Options.get_io_engine
        elif key == "GFO_ON_DATA_ERROR":
            result = Options.get_on_data_error
        elif key == "GFO_REMOVE_TEMP_FILES":
            result = Options.get_remove_temp_files
        elif key == "GFO_WORKER_TYPE":
            result = Options.get_worker_type
        else:
            raise ValueError(f"Unexpected key: {key}")

    assert result == expected


@pytest.mark.parametrize(
    "key, invalid_value, expected_error",
    [
        ("GFO_IO_ENGINE", "invalid", "invalid value for configoption <GFO_IO_ENGINE>"),
        (
            "GFO_ON_DATA_ERROR",
            "invalid",
            "invalid value for configoption <GFO_ON_DATA_ERROR>",
        ),
        (
            "GFO_REMOVE_TEMP_FILES",
            "invalid",
            "invalid value for bool configoption <GFO_REMOVE_TEMP_FILES>",
        ),
        (
            "GFO_SLIVER_TOLERANCE",
            "not_a_number",
            "invalid value for configoption <GFO_SLIVER_TOLERANCE>",
        ),
        (
            "GFO_TMPDIR",
            "   ",
            "GFO_TMPDIR='' environment variable found which is not supported",
        ),
        (
            "GFO_WORKER_TYPE",
            "invalid",
            "invalid value for configoption <GFO_WORKER_TYPE>",
        ),
    ],
)
def test_configoptions_invalid(key, invalid_value, expected_error):
    with (
        gfo.TempEnv({key: invalid_value}),
        pytest.raises(ValueError, match=expected_error),
    ):
        if key == "GFO_IO_ENGINE":
            _ = Options.get_io_engine
        elif key == "GFO_ON_DATA_ERROR":
            _ = Options.get_on_data_error
        elif key == "GFO_REMOVE_TEMP_FILES":
            _ = Options.get_remove_temp_files
        elif key == "GFO_SLIVER_TOLERANCE":
            _ = Options.sliver_tolerance
        elif key == "GFO_TMPDIR":
            _ = Options.get_tmp_dir
        elif key == "GFO_WORKER_TYPE":
            _ = Options.get_worker_type
        else:
            raise ValueError(f"Unexpected key: {key}")


@pytest.mark.parametrize(
    "tolerance, crs, expected",
    [
        ("0.1", None, 0.1),
        ("0.2", None, 0.2),
        (None, None, 0.0),
        (None, CRS.from_epsg(31370), 0.001),
        (None, CRS.from_epsg(3857), 0.001),
        (None, CRS.from_epsg(2277), 0.001),  # CRS in feet -> same tolerance
        (None, CRS.from_epsg(4326), 1e-7),
        ("0.5", CRS.from_epsg(31370), 0.5),
        ("-0.5", CRS.from_epsg(4326), -0.5),
    ],
)
def test_configoptions_sliver_tolerance(tolerance, crs, expected):
    """Test ConfigOptions.sliver_tolerance method."""
    with gfo.TempEnv({"GFO_SLIVER_TOLERANCE": tolerance}):
        result = _configoptions_helper.ConfigOptions.sliver_tolerance(crs)
        assert result == expected


def test_configoptions_tmpdir(tmp_path):
    """Test ConfigOptions.tmp_dir property."""
    with gfo.TempEnv({"GFO_TMPDIR": str(tmp_path)}):
        assert Options.get_tmp_dir == tmp_path

    # If GFO_TMPDIR is not set, a "geofileops" subdirectory in the system temp dir is
    # used.
    with gfo.TempEnv({"GFO_TMPDIR": None}):
        tmp_dir = Options.get_tmp_dir
        assert tmp_dir.exists()
        assert tmp_dir.name == "geofileops"
        tempdir = tempfile.gettempdir()
        assert str(tmp_dir).startswith(tempdir)
