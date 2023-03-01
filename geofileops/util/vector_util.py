import warnings

warnings.warn(
    "vector_util is deprecated, import grid_util and/or geometry_util directly instead",
    FutureWarning,
)
from .grid_util import *  # noqa: F401, F403
from .geometry_util import *  # noqa: F401, F403
