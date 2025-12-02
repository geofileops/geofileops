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
