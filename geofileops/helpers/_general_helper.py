"""General helper functions, specific for geofileops."""

import os
import tempfile
from pathlib import Path

from geofileops.helpers._configoptions_helper import ConfigOptions
from geofileops.util import _io_util


def create_gfo_tmp_dir(base_dirname: str, parent_dir: Path | None = None) -> Path:
    """Create a temporary directory for geofileops operations.

    Args:
        base_dirname (str): The base name of the temporary directory to create. The
            following characters are replaced to "_": "/", " ".
        parent_dir (Path | None, optional): The parent directory to create the
            temporary directory in. If None, the directory specified in the environment
            variable `GFO_TMPDIR` is used. If that  that does not exist, a "geofileops"
            subdirectory in :func:`tempfile.gettempdir` is used. Defaults to None.

    Returns:
        Path: The path to the created temporary directory.
    """
    if parent_dir is None:
        parent_dir = get_gfo_tmp_dir()
    base_dirname = base_dirname.replace("/", "_").replace(" ", "_")

    return _io_util.create_tempdir(base_dirname, parent_dir)


def get_gfo_tmp_dir() -> Path:
    """Get the base temporary directory for geofileops operations.

    This is either the directory specified in the environment variable `GFO_TMPDIR` or
    a subdirectory "geofileops" in :func:`tempfile.gettempdir`.

    Returns:
        Path: The base temporary directory for geofileops operations.
    """
    # Check if a custom temp dir is specified in environment variable GFO_TMPDIR.
    tmp_dir = os.environ.get("GFO_TMPDIR")
    if tmp_dir is None:
        gfo_tmp_dir = Path(tempfile.gettempdir()) / "geofileops"
    elif tmp_dir != "":
        gfo_tmp_dir = Path(tmp_dir)
    else:
        raise RuntimeError(
            "GFO_TMPDIR='' environment variable found which is not supported."
        )

    # Make sure the dir exists
    gfo_tmp_dir.mkdir(parents=True, exist_ok=True)

    return gfo_tmp_dir


def worker_type_to_use(input_layer_featurecount: int) -> str:
    worker_type = ConfigOptions.worker_type
    if worker_type in ("threads", "processes"):
        return worker_type

    # Processing in threads is 2x faster for small datasets
    if input_layer_featurecount <= 100:
        return "threads"

    return "processes"
