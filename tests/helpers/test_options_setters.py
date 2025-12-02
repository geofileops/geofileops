"""Test functions for the setters of geofileops options."""

import os

import geofileops as gfo


def test_copy_layer_sqlite_direct() -> None:
    """Test the copy_layer_sqlite_direct option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_COPY_LAYER_SQLITE_DIRECT"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.copy_layer_sqlite_direct(False)
    assert os.environ[key] == "FALSE"

    # Test setting the option temporarily using context manager
    with gfo.options.copy_layer_sqlite_direct(True):
        assert os.environ[key] == "TRUE"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was False)
    assert os.environ[key] == "FALSE"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.copy_layer_sqlite_direct(True):
        assert os.environ[key] == "TRUE"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ


def test_io_engine() -> None:
    """Test the io_engine option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_IO_ENGINE"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.io_engine("fiona")
    assert os.environ[key] == "FIONA"

    # Test setting the option temporarily using context manager
    with gfo.options.io_engine("pyogrio"):
        assert os.environ[key] == "PYOGRIO"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was "fiona")
    assert os.environ[key] == "FIONA"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.io_engine("pyogrio-arrow"):
        assert os.environ[key] == "PYOGRIO-ARROW"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ


def test_on_data_error() -> None:
    """Test the on_data_error option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_ON_DATA_ERROR"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.on_data_error("warn")
    assert os.environ[key] == "WARN"

    # Test setting the option temporarily using context manager
    with gfo.options.on_data_error("raise"):
        assert os.environ[key] == "RAISE"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was "warn")
    assert os.environ[key] == "WARN"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.on_data_error("raise"):
        assert os.environ[key] == "RAISE"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ


def test_remove_temp_files() -> None:
    """Test the remove_temp_files option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_REMOVE_TEMP_FILES"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.remove_temp_files(False)
    assert os.environ[key] == "FALSE"

    # Test setting the option temporarily using context manager
    with gfo.options.remove_temp_files(True):
        assert os.environ[key] == "TRUE"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was False)
    assert os.environ[key] == "FALSE"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.remove_temp_files(True):
        assert os.environ[key] == "TRUE"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ


def test_tmp_dir() -> None:
    """Test the tmp_dir option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_TMPDIR"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.tmp_dir("/tmp/geofileops_test")
    assert os.environ[key] == "/tmp/geofileops_test"

    # Test setting the option temporarily using context manager
    with gfo.options.tmp_dir("/tmp/geofileops_temp"):
        assert os.environ[key] == "/tmp/geofileops_temp"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was "/tmp/geofileops_test")
    assert os.environ[key] == "/tmp/geofileops_test"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.tmp_dir("/tmp/geofileops_temp2"):
        assert os.environ[key] == "/tmp/geofileops_temp2"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ


def test_worker_type() -> None:
    """Test the worker_type option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_WORKER_TYPE"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.worker_type("threads")
    assert os.environ[key] == "THREADS"

    # Test setting the option temporarily using context manager
    with gfo.options.worker_type("processes"):
        assert os.environ[key] == "PROCESSES"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was "threads")
    assert os.environ[key] == "THREADS"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.worker_type("auto"):
        assert os.environ[key] == "AUTO"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ


def test_subdivide_check_parallel_fraction() -> None:
    """Test the subdivide_check_parallel_fraction option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_SUBDIVIDE_CHECK_PARALLEL_FRACTION"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.subdivide_check_parallel_fraction(10)
    assert os.environ[key] == "10"

    # Test setting the option temporarily using context manager
    with gfo.options.subdivide_check_parallel_fraction(5):
        assert os.environ[key] == "5"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was 10)
    assert os.environ[key] == "10"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.subdivide_check_parallel_fraction(20):
        assert os.environ[key] == "20"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ


def test_subdivide_check_parallel_rows() -> None:
    """Test the subdivide_check_parallel_rows option setter."""
    # Make sure the environment variable is not set at the start of the test
    key = "GFO_SUBDIVIDE_CHECK_PARALLEL_ROWS"
    if key in os.environ:
        del os.environ[key]

    # Test setting the option permanently
    gfo.options.subdivide_check_parallel_rows(100000)
    assert os.environ[key] == "100000"

    # Test setting the option temporarily using context manager
    with gfo.options.subdivide_check_parallel_rows(50000):
        assert os.environ[key] == "50000"

    # After exiting the context manager, the value should be restored to the last
    # permanent setting (which was 100000)
    assert os.environ[key] == "100000"

    # Clean up by removing the environment variable
    del os.environ[key]

    # Test setting the option temporarily using context manager
    with gfo.options.subdivide_check_parallel_rows(200000):
        assert os.environ[key] == "200000"

    # After exiting the context manager, the environment variable should be removed
    assert key not in os.environ
