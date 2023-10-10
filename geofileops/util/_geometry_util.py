"""
Module containing utilities regarding low level vector operations.
"""

import enum
import logging
import warnings

from geofileops import GeometryType

# Get a logger...
logger = logging.getLogger(__name__)


class BufferJoinStyle(enum.Enum):
    """
    Enumeration of the available buffer styles for intermediate points.

    Relevant for the end points of a line or polygon geometry.
    """

    ROUND = 1
    MITRE = 2
    BEVEL = 3


class BufferEndCapStyle(enum.Enum):
    """
    Enumeration of the possible end point buffer styles.

    Relevant for the end points of a line or point geometry.
    """

    ROUND = 1
    FLAT = 2
    SQUARE = 3


class SimplifyAlgorithm(enum.Enum):
    """
    Enumeration of the supported simplification algorythms.
    """

    RAMER_DOUGLAS_PEUCKER = "rdp"
    LANG = "lang"
    LANGP = "lang+"
    VISVALINGAM_WHYATT = "vw"


def to_multi_type(geometrytypename: str) -> str:
    """
    Map the input geometry type to the corresponding 'MULTI' geometry type...

    DEPRECATED, use to_multigeometrytype

    Args:
        geometrytypename (str): Input geometry type

    Raises:
        ValueError: If input geometrytype is not known.

    Returns:
        str: Corresponding 'MULTI' geometry type
    """
    warnings.warn(
        "to_generaltypeid is deprecated, use GeometryType.to_multigeometrytype",
        FutureWarning,
        stacklevel=2,
    )
    return GeometryType(geometrytypename).to_multitype.name


def to_generaltypeid(geometrytypename: str) -> int:
    """
    Map the input geometry type name to the corresponding geometry type id.

    Possible valuesh:
        * 1 = POINT-type
        * 2 = LINESTRING-type
        * 3 = POLYGON-type

    DEPRECATED, use to_primitivetypeid()

    Args:
        geometrytypename (str): Input geometry type

    Raises:
        ValueError: If input geometrytype is not known.

    Returns:
        int: Corresponding geometry type id
    """
    warnings.warn(
        "to_generaltypeid is deprecated, use GeometryType.to_primitivetypeid",
        FutureWarning,
        stacklevel=2,
    )
    return GeometryType(geometrytypename).to_primitivetype.value
