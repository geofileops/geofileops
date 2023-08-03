import pytest
import shapely

from geofileops.util import _sqlite_userdefined as sqlite_userdefined


def test_st_difference_collection():
    # None parameter for geom or geom_to_subtract -> just returns input!
    assert (
        sqlite_userdefined.st_difference_collection(shapely.Point().wkb, None)
        == shapely.Point().wkb
    )
    assert sqlite_userdefined.st_difference_collection(None, None) is None
    point11 = shapely.Point(1, 1)
    assert sqlite_userdefined.st_difference_collection(None, point11.wkb) is None
    assert sqlite_userdefined.st_difference_collection(point11.wkb, None) == point11.wkb
    assert sqlite_userdefined.st_difference_collection(point11.wkb, None) == point11.wkb

    # Difference of polygons gives a polygon
    box0_5 = shapely.box(0, 0, 5, 5)
    box0_10 = shapely.box(0, 0, 10, 10)
    assert (
        sqlite_userdefined.st_difference_collection(box0_10.wkb, box0_5.wkb)
        == shapely.difference(box0_10, box0_5).wkb
    )
    # Difference of polygons gives a linestring to test -> No testcase at hand
    assert (
        sqlite_userdefined.st_difference_collection(
            box0_10.wkb, box0_5.wkb, keep_geom_type=1
        )
        == shapely.difference(box0_10, box0_5).wkb
    )

    # Result of the difference is None.
    assert sqlite_userdefined.st_difference_collection(point11.wkb, box0_10.wkb) is None


def test_st_difference_collection_invalid_params():
    # geom_to_subtract is not a wkb
    with pytest.raises(TypeError, match="Expected bytes or string, got Point"):
        sqlite_userdefined.st_difference_collection(
            shapely.Point(1, 1).wkb, shapely.Point(1, 1)
        )
    # geom is not a wkb
    with pytest.raises(TypeError, match="Expected bytes or string, got Point"):
        sqlite_userdefined.st_difference_collection(
            shapely.Point(1, 1), shapely.Point(1, 1).wkb
        )

    # keep_geom_type should be int (0 or 1)
    with pytest.raises(TypeError, match="keep_geom_type must be int"):
        sqlite_userdefined.st_difference_collection(
            shapely.Point(1, 1).wkb, shapely.Point(1, 1).wkb, keep_geom_type="TRUE"
        )
    with pytest.raises(ValueError, match="keep_geom_type has invalid value"):
        sqlite_userdefined.st_difference_collection(
            shapely.Point(1, 1).wkb, shapely.Point(1, 1).wkb, keep_geom_type=5
        )
