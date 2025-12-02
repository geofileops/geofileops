"""Module with convenience functions to set the existing geofileops `Options`."""

# This module makes it possible to only make the setters public + shortens the names in
# the public API.

from geofileops.helpers._options import ConfigOptions


class options:
    """Helper function to make it easy to set geofileops runtime options.

    The runtime options are saved to and read from environment variables.

    All functions can be used in two ways:
        1. Permanently set the option by calling the function directly.
        2. Temporarily set the option by using the function as a context manager.
    """

    set_copy_layer_sqlite_direct = ConfigOptions.set_copy_layer_sqlite_direct

    set_io_engine = ConfigOptions.set_io_engine
    set_on_data_error = ConfigOptions.set_on_data_error
    set_remove_temp_files = ConfigOptions.set_remove_temp_files
    set_subdivide_check_parallel_rows = ConfigOptions.set_subdivide_check_parallel_rows
    set_subdivide_check_parallel_fraction = (
        ConfigOptions.set_subdivide_check_parallel_fraction
    )
    set_tmp_dir = ConfigOptions.set_tmp_dir
    set_worker_type = ConfigOptions.set_worker_type
