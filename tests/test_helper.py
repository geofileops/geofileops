# -*- coding: utf-8 -*-
"""
Helper functions for all tests.
"""

from pathlib import Path
import tempfile
from typing import List, Optional, Union

import geopandas as gpd
import geopandas._compat as gpd_compat
import geopandas.testing as gpd_testing

if gpd_compat.USE_PYGEOS:
    import pygeos as shapely2_or_pygeos
else:
    import shapely as shapely2_or_pygeos
import shapely.geometry as sh_geom

import geofileops as gfo
from geofileops.util import geodataframe_util
from geofileops.util import geoseries_util

_data_dir = Path(__file__).parent.resolve() / "data"
EPSGS = [31370, 4326]
GRIDSIZE_DEFAULT = 0.0
SUFFIXES = [".gpkg", ".shp"]
TESTFILES = ["polygon-parcel", "linestring-row-trees", "point"]
WHERE_AREA_GT_400 = "ST_Area({geometrycolumn}) > 400"
WHERE_AREA_GT_5000 = "ST_Area({geometrycolumn}) > 5000"
WHERE_LENGTH_GT_1000 = "ST_Length({geometrycolumn}) > 1000"
WHERE_LENGTH_GT_200000 = "ST_Length({geometrycolumn}) > 200000"


def prepare_expected_result(
    gdf: gpd.GeoDataFrame,
    keep_empty_geoms: bool,
    gridsize: float = 0.0,
    where: Optional[str] = None,
    explodecollections=False,
    columns: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    """Prepare expected data"""

    expected_gdf = gdf.copy()

    if gridsize != 0.0:
        expected_gdf.geometry = shapely2_or_pygeos.set_precision(
            expected_gdf.geometry.array.data, grid_size=gridsize
        )
    if explodecollections:
        expected_gdf = expected_gdf.explode(ignore_index=True)

    # Check what filtering is needed
    filter_area_gt = None
    if where is None or where == "":
        pass
    elif where == "ST_Area({geometrycolumn}) > 400":
        filter_area_gt = 400
    elif where == "ST_Area({geometrycolumn}) > 5000":
        filter_area_gt = 5000
    else:
        raise ValueError(f"unsupported where parameter in test: {where}")

    # Apply filtering
    if keep_empty_geoms is None or not keep_empty_geoms:
        expected_gdf = expected_gdf[~expected_gdf.geometry.isna()]
        expected_gdf = expected_gdf[~expected_gdf.geometry.is_empty]
    if filter_area_gt is not None:
        expected_gdf = expected_gdf[expected_gdf.geometry.area > filter_area_gt]

    if columns is not None:
        column_mapper = {}
        columns_to_drop = []
        columns_dict = {column.upper(): column for column in columns}
        for column in expected_gdf.columns:
            if column.upper() in columns_dict:
                column_mapper[column] = columns_dict[column.upper()]
            elif column != "geometry":
                columns_to_drop.append(column)
        expected_gdf = expected_gdf.rename(columns=column_mapper)
        if len(columns_to_drop) > 0:
            expected_gdf = expected_gdf.drop(columns=columns_to_drop)

    assert isinstance(expected_gdf, gpd.GeoDataFrame)
    return expected_gdf


def prepare_test_file(
    input_path: Path,
    output_dir: Path,
    suffix: str,
    crs_epsg: Optional[int] = None,
    use_cachedir: bool = False,
) -> Path:
    # Tmp dir
    if use_cachedir is True:
        tmp_cache_dir = Path(tempfile.gettempdir()) / "geofileops_test_data"
        tmp_cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_cache_dir = output_dir

    # If crs_epsg specified and test input file in wrong crs_epsg, reproject
    input_prepared_path = input_path
    if crs_epsg is not None:
        input_prepared_path = tmp_cache_dir / f"{input_path.stem}_{crs_epsg}{suffix}"
        if input_prepared_path.exists() is False:
            input_layerinfo = gfo.get_layerinfo(input_path)
            assert input_layerinfo.crs is not None
            if input_layerinfo.crs.to_epsg() == crs_epsg:
                if input_path.suffix == suffix:
                    gfo.copy(input_path, input_prepared_path)
                else:
                    gfo.copy_layer(input_path, input_prepared_path)
            else:
                test_gdf = gfo.read_file(input_path)
                test_gdf = test_gdf.to_crs(crs_epsg)
                assert isinstance(test_gdf, gpd.GeoDataFrame)
                gfo.to_file(test_gdf, input_prepared_path)
    elif input_path.suffix != suffix:
        # No crs specified, but different suffix asked, so convert file
        input_prepared_path = tmp_cache_dir / f"{input_path.stem}{suffix}"
        if input_prepared_path.exists() is False:
            gfo.copy_layer(input_path, input_prepared_path)

    # Now copy the prepared file to the output dir
    output_path = output_dir / input_prepared_path.name
    if str(input_prepared_path) != str(output_path):
        gfo.copy(input_prepared_path, output_path)
    return output_path


def get_testfile(
    testfile: str,
    dst_dir: Optional[Path] = None,
    suffix: str = ".gpkg",
    epsg: int = 31370,
    empty: bool = False,
) -> Path:
    # Prepare original filepath
    testfile_path = _data_dir / f"{testfile}.gpkg"
    if testfile_path.exists is False:
        raise ValueError(f"Invalid testfile type: {testfile}")
    if dst_dir is None:
        dst_dir = Path(tempfile.gettempdir()) / "geofileops_test_data"
    assert isinstance(dst_dir, Path)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Prepare file + return
    empty_str = "_empty" if empty else ""
    prepared_path = dst_dir / f"{testfile_path.stem}_{epsg}{empty_str}{suffix}"
    dst_geofiletype = gfo.GeofileType(prepared_path)
    if prepared_path.exists():
        return prepared_path
    layers = gfo.listlayers(testfile_path)
    if len(layers) > 1 and dst_geofiletype.is_singlelayer:
        raise ValueError(
            f"multilayer testfile ({testfile}) cannot be converted to single layer "
            f"geofiletype: {dst_geofiletype}"
        )

    # Convert all layers found
    for layer in layers:
        gfo.copy_layer(
            testfile_path,
            prepared_path,
            src_layer=layer,
            dst_layer=layer,
            dst_crs=epsg,
            reproject=True,
            append=True,
            preserve_fid=True,
        )
        # If all rows need to be deleted
        if empty:
            # Remove all rows from destination layer.
            # GDAL only supports DELETE using SQLITE dialect, not with OGRSQL.
            if dst_geofiletype.is_singlelayer:
                # Layer name can be different for singlelayer output files
                layer = gfo.listlayers(prepared_path)[0]
            gfo.execute_sql(
                prepared_path,
                sql_stmt=f'DELETE FROM "{layer}"',
                sql_dialect="SQLITE",
            )

    return prepared_path


class TestData:
    crs_epsg = 31370
    point = sh_geom.Point((0, 0))
    multipoint = sh_geom.MultiPoint([(0, 0), (10, 10), (20, 20)])
    linestring = sh_geom.LineString([(0, 0), (10, 10), (20, 20)])
    multilinestring = sh_geom.MultiLineString(
        [linestring.coords, [(100, 100), (110, 110), (120, 120)]]
    )
    polygon_with_island = sh_geom.Polygon(
        shell=[(0.01, 0), (0.01, 10), (1, 10), (10, 10), (10, 0), (0.01, 0)],
        holes=[[(2, 2), (2, 8), (8, 8), (8, 2), (2, 2)]],
    )
    polygon_no_islands = sh_geom.Polygon(
        shell=[(100.01, 100), (100.01, 110), (110, 110), (110, 100), (100.01, 100)]
    )
    polygon_with_island2 = sh_geom.Polygon(
        shell=[(20, 20), (20, 30), (21, 30), (30, 30), (30, 20), (20, 20)],
        holes=[[(22, 22), (22, 28), (28, 28), (28, 22), (22, 22)]],
    )
    multipolygon = sh_geom.MultiPolygon([polygon_no_islands, polygon_with_island2])
    geometrycollection = sh_geom.GeometryCollection(
        [
            point,
            multipoint,
            linestring,
            multilinestring,
            polygon_with_island,
            multipolygon,
        ]
    )
    polygon_small_island = sh_geom.Polygon(
        shell=[(40, 40), (40, 50), (41, 50), (50, 50), (50, 40), (40, 40)],
        holes=[[(42, 42), (42, 43), (43, 43), (43, 42), (42, 42)]],
    )


def create_tempdir(base_dirname: str, parent_dir: Optional[Path] = None) -> Path:
    # Parent
    if parent_dir is None:
        parent_dir = Path(tempfile.gettempdir())

    for i in range(1, 999999):
        try:
            tempdir = parent_dir / f"{base_dirname}_{i:06d}"
            tempdir.mkdir(parents=True)
            return Path(tempdir)
        except FileExistsError:
            continue

    raise Exception(
        "Wasn't able to create a temporary dir with basedir: "
        f"{parent_dir / base_dirname}"
    )


def assert_geodataframe_equal(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    check_dtype=True,
    check_index_type: Union[bool, str] = "equiv",
    check_column_type: Union[bool, str] = "equiv",
    check_frame_type=True,
    check_like=False,
    check_less_precise=False,
    check_geom_type=False,
    check_geom_empty_vs_None=True,
    check_crs=True,
    normalize=False,
    promote_to_multi=False,
    sort_columns=False,
    sort_values=False,
    output_dir: Optional[Path] = None,
):
    """
    Check that two GeoDataFrames are equal/

    Parameters
    ----------
    left, right : two GeoDataFrames
    check_dtype : bool, default True
        Whether to check the DataFrame dtype is identical.
    check_index_type, check_column_type : bool, default 'equiv'
        Check that index types are equal.
    check_frame_type : bool, default True
        Check that both are same type (*and* are GeoDataFrames). If False,
        will attempt to convert both into GeoDataFrame.
    check_like : bool, default False
        If true, ignore the order of rows & columns
    check_less_precise : bool, default False
        If True, use geom_almost_equals. if False, use geom_equals.
    check_geom_type : bool, default False
        If True, check that all the geom types are equal.
    check_geom_empty_vs_None : bool, default True
        If False, ignore differences between empty and None geometries.
    check_crs: bool, default True
        If `check_frame_type` is True, then also check that the
        crs matches.
    normalize: bool, default False
        If True, normalize the geometries before comparing equality.
        Typically useful with ``check_less_precise=True``, which uses
        ``geom_almost_equals`` and requires exact coordinate order.
    promote_to_multi: bool, default False
        If True, promotes to multi.
    sort_columns: bool, default False
        If True, sort the columns of the dataframe before compare.
    sort_values: bool, default False
        If True, sort the values of the geodataframe, including the geometry
        (as WKT).
    output_dir: Path, default None
        If not None, the left and right dataframes will be written to the
        directory as geojson files. If normalize, promote_to_multi and/or
        sort_values are True, the will be applied before writing.
    """
    if sort_columns:
        left = left[sorted(left.columns)]
        right = right[sorted(right.columns)]

    if not check_geom_empty_vs_None:
        # Set empty geoms to None for both inputs
        left = left.copy()
        left.loc[left.geometry.is_empty, ["geometry"]] = None
        right = right.copy()
        right.loc[right.geometry.is_empty, ["geometry"]] = None

    if sort_values:
        if normalize:
            left.geometry = gpd.GeoSeries(
                shapely2_or_pygeos.normalize(left.geometry.array.data), index=left.index
            )
            right.geometry = gpd.GeoSeries(
                shapely2_or_pygeos.normalize(right.geometry.array.data),
                index=right.index,
            )
        if promote_to_multi:
            left.geometry = geoseries_util.harmonize_geometrytypes(
                left.geometry, force_multitype=True
            )
            right.geometry = geoseries_util.harmonize_geometrytypes(
                right.geometry, force_multitype=True
            )
        left = geodataframe_util.sort_values(left).reset_index(drop=True)
        right = geodataframe_util.sort_values(right).reset_index(drop=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "left.geojson"
        gfo.to_file(left, output_path, create_spatial_index=None)
        output_path = output_dir / "right.geojson"
        gfo.to_file(right, output_path, create_spatial_index=None)

    gpd_testing.assert_geodataframe_equal(
        left=left,
        right=right,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_frame_type=check_frame_type,
        check_like=check_like,
        check_less_precise=check_less_precise,
        check_geom_type=check_geom_type,
        check_crs=check_crs,
        normalize=normalize,
    )
