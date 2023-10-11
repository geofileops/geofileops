"""
Tests for functionalities in geofiletype.
"""

from pathlib import Path

import pytest

import geofileops as gfo
from geofileops.util import _geofileinfo
from geofileops.util._geofileinfo import GeofileType
from tests import test_helper


def test_geofiletype_enum():
    # Test ESRIShapefile geofiletype
    # Test getting a geofiletype for a suffix
    assert GeofileType(".shp") == GeofileType.ESRIShapefile
    assert GeofileType(".shp").ogrdriver == "ESRI Shapefile"

    # Test getting a geofiletype for a Path (case insensitive)
    path = Path("/testje/path_naar_file.sHp")
    assert GeofileType(path) == GeofileType.ESRIShapefile

    # Test getting a geofiletype for an ogr driver
    assert GeofileType("ESRI Shapefile") == GeofileType.ESRIShapefile

    # GPKG geofiletype
    # Test getting a geofiletype for a suffix
    assert GeofileType(".gpkg") == GeofileType.GPKG
    assert GeofileType(".gpkg").ogrdriver == "GPKG"

    # Test getting a geofiletype for a Path (case insensitive)
    path = Path("/testje/path_naar_file.gPkG")
    assert GeofileType(path) == GeofileType.GPKG

    # Test getting a geofiletype for an ogr driver
    assert GeofileType("GPKG") == GeofileType.GPKG

    # SQLite geofiletype
    # Test getting a geofiletype for a suffix
    assert GeofileType(".sqlite") == GeofileType.SQLite

    # Test getting a geofiletype for a Path (case insensitive)
    path = Path("/testje/path_naar_file.sQlItE")
    assert GeofileType(path) == GeofileType.SQLite

    # Test getting a geofiletype for an ogr driver
    assert GeofileType("SQLite") == GeofileType.SQLite


@pytest.mark.parametrize(
    "suffix, driver",
    [(".gpkg", "GPKG"), (".GPKG", "GPKG"), (".shp", "ESRI Shapefile"), (".csv", "CSV")],
)
@pytest.mark.parametrize(
    "existing_file, invalid_file", [(True, True), (True, False), (False, False)]
)
def test_get_driver(tmp_path, suffix, driver, existing_file: bool, invalid_file: bool):
    """Get a driver."""
    # Prepare test data
    if existing_file:
        if invalid_file:
            test_path = tmp_path / f"test_invalid{suffix}"
            test_path.touch()
        else:
            test_path = test_helper.get_testfile("polygon-parcel", suffix=suffix)
    else:
        test_path = tmp_path / f"test_unexisting{suffix}"

    assert gfo.get_driver(test_path) == driver


def test_get_geofileinfo():
    # Test ESRIShapefile geofiletype
    # Test getting a geofiletype for a suffix
    info = _geofileinfo.get_geofileinfo(".shp")
    assert info.driver == "ESRI Shapefile"
    assert info.is_fid_zerobased
    assert info.is_singlelayer
    assert not info.is_spatialite_based

    # GPKG geofiletype
    # Test getting a geofiletype for a suffix
    info = _geofileinfo.get_geofileinfo(".gpKG")
    assert info.driver == "GPKG"
    assert not info.is_fid_zerobased
    assert not info.is_singlelayer
    assert info.is_spatialite_based

    # SQLite geofiletype
    # Test getting a geofiletype for a suffix
    info = _geofileinfo.get_geofileinfo(".sqlite")
    assert info.driver == "SQLite"
    assert not info.is_fid_zerobased
    assert not info.is_singlelayer
    assert info.is_spatialite_based

    # CSV geofiletype
    # Test getting a geofiletype for a suffix
    info = _geofileinfo.get_geofileinfo(".csv")
    assert info.driver == "CSV"
    assert not info.is_fid_zerobased
    assert info.is_singlelayer
    assert not info.is_spatialite_based
