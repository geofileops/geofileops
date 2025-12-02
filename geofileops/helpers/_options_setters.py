"""Module with convenience functions to set the existing geofileops `Options`."""

# This module makes it possible to only make the setters public + shortens the names in
# the public API.

from geofileops.helpers._options import Options

copy_layer_sqlite_direct = Options.set_copy_layer_sqlite_direct
io_engine = Options.set_io_engine
on_data_error = Options.set_on_data_error
remove_temp_files = Options.set_remove_temp_files
subdivide_check_parallel_rows = Options.set_subdivide_check_parallel_rows
subdivide_check_parallel_fraction = Options.set_subdivide_check_parallel_fraction
tmp_dir = Options.set_tmp_dir
worker_type = Options.set_worker_type
