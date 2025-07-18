"""
Helper functions for all tests.
"""

import os
import re
import tempfile
from pathlib import Path
from stat import S_IRGRP, S_IROTH, S_IRUSR, S_IRWXG, S_IRWXO, S_IRWXU

import geopandas as gpd
import geopandas.testing as gpd_testing
import shapely
import shapely.geometry as sh_geom

import geofileops as gfo
from geofileops.util import (
    _geofileinfo,
    _geopath_util,
    _geoseries_util,
    _io_util,
    geodataframe_util,
)

try:
    import matplotlib.colors as mcolors
    from matplotlib import figure as mpl_figure
except ImportError:
    mcolors = None
    mpl_figure = None

data_dir = Path(__file__).parent.resolve() / "data"
data_url = "https://raw.githubusercontent.com/geofileops/geofileops/main/tests/data"

EPSGS = [31370, 4326]
GRIDSIZE_DEFAULT = 0.0
SUFFIXES_FILEOPS = [".gpkg", ".shp", ".csv"]
SUFFIXES_FILEOPS_EXT = [".gpkg", ".gpkg.zip", ".shp.zip", ".shp", ".csv"]
SUFFIXES_GEOOPS = [".gpkg", ".shp"]
SUFFIXES_GEOOPS_INPUT = [".gpkg", ".gpkg.zip", ".shp.zip", ".shp"]
TESTFILES = ["polygon-parcel", "linestring-row-trees", "point"]
WHERE_AREA_GT_400 = "ST_Area({geometrycolumn}) > 400"
WHERE_AREA_GT_5000 = "ST_Area({geometrycolumn}) > 5000"
WHERE_LENGTH_GT_1000 = "ST_Length({geometrycolumn}) > 1000"
WHERE_LENGTH_GT_200000 = "ST_Length({geometrycolumn}) > 200000"

RUNS_LOCAL = True
if "GITHUB_ACTIONS" in os.environ:
    RUNS_LOCAL = False


def prepare_expected_result(
    gdf: gpd.GeoDataFrame,
    keep_empty_geoms: bool,
    gridsize: float = 0.0,
    where_post: str | None = None,
    explodecollections=False,
    columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Prepare expected data"""
    if keep_empty_geoms is None:
        raise ValueError("keep_empty_geoms cannot be None")

    exp_gdf = gdf.copy()

    if gridsize != 0.0:
        exp_gdf.geometry = shapely.set_precision(exp_gdf.geometry, grid_size=gridsize)
    if explodecollections:
        exp_gdf = exp_gdf.explode(ignore_index=True)

    # Check what filtering is needed
    filter_area_gt = None
    if where_post is None or where_post == "":
        pass
    elif where_post == "ST_Area({geometrycolumn}) > 400":
        filter_area_gt = 400
    elif where_post == "ST_Area({geometrycolumn}) > 5000":
        filter_area_gt = 5000
    else:
        raise ValueError(f"unsupported where_post parameter in test: {where_post}")

    # Apply filtering
    # If keep_empty_geoms is None, treat it as False as this is the default value
    if not keep_empty_geoms:
        exp_gdf = exp_gdf[~exp_gdf.geometry.isna()]
        exp_gdf = exp_gdf[~exp_gdf.geometry.is_empty]
    if filter_area_gt is not None:
        exp_gdf = exp_gdf[exp_gdf.geometry.area > filter_area_gt]

    if columns is not None:
        column_mapper = {}
        columns_to_drop = []
        columns_dict = {column.upper(): column for column in columns}
        for column in exp_gdf.columns:
            if column.upper() in columns_dict:
                column_mapper[column] = columns_dict[column.upper()]
            elif column != "geometry":
                columns_to_drop.append(column)
        exp_gdf = exp_gdf.rename(columns=column_mapper)
        if len(columns_to_drop) > 0:
            exp_gdf = exp_gdf.drop(columns=columns_to_drop)

    assert isinstance(exp_gdf, gpd.GeoDataFrame)
    return exp_gdf


def prepare_test_file(
    input_path: Path,
    output_dir: Path,
    suffix: str,
    crs_epsg: int | None = None,
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
    dst_dir: Path | None = None,
    suffix: str = ".gpkg",
    epsg: int = 31370,
    empty: bool = False,
    dimensions: str | None = None,
    explodecollections: bool = False,
    read_only: bool | None = None,
) -> Path:
    if dst_dir is None:
        read_only = True

    prepared_path = _get_testfile(
        testfile=testfile,
        dst_dir=dst_dir,
        suffix=suffix,
        epsg=epsg,
        empty=empty,
        dimensions=dimensions,
        explodecollections=explodecollections,
    )

    # Make input read-only
    if read_only:
        set_read_only(prepared_path, read_only=True)

    return prepared_path


def _get_testfile(
    testfile: str,
    dst_dir: Path | None = None,
    suffix: str = ".gpkg",
    epsg: int = 31370,
    empty: bool = False,
    dimensions: str | None = None,
    explodecollections: bool = False,
) -> Path:
    # Prepare original filepath.
    testfile_path = data_dir / f"{testfile}.gpkg"
    if not testfile_path.exists():
        raise ValueError(f"Invalid testfile type: {testfile}")

    # Prepare destination location
    if dst_dir is None:
        dst_dir = Path(tempfile.gettempdir()) / "geofileops_test_data"
    assert isinstance(dst_dir, Path)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Prepare file + return
    empty_str = "_empty" if empty else ""
    prepared_path = (
        dst_dir / f"{testfile_path.stem}_{epsg}_{dimensions}{empty_str}{suffix}"
    )
    if prepared_path.exists():
        return prepared_path

    # Test file doesn't exist yet, so create it
    # To be safe for parallelized tests, lock the creation.
    prepared_lock_path = Path(f"{prepared_path}.lock")
    try:
        _io_util.create_file_atomic_wait(
            prepared_lock_path, time_between_attempts=0.1, timeout=60
        )

        # Make sure it wasn't created by another process while waiting for the lock file
        if prepared_path.exists():
            return prepared_path

        # Prepare the file in a tmp file so the file is not visible to other
        # processes until it is completely ready.
        tmp_stem = f"{_geopath_util.stem(prepared_path)}_tmp"
        tmp_path = _geopath_util.with_stem(prepared_path, tmp_stem)
        layers = gfo.listlayers(testfile_path)
        dst_info = _geofileinfo.get_geofileinfo(tmp_path)
        if len(layers) > 1 and dst_info.is_singlelayer:
            raise ValueError(
                f"multilayer testfile ({testfile}) cannot be converted to single layer "
                f"geofiletype: {dst_info.driver}"
            )

        # Convert all layers found
        for src_layer in layers:
            # Single layer files have stem as layername
            assert isinstance(tmp_path, Path)
            if dst_info.is_singlelayer:
                dst_layer = tmp_stem
                preserve_fid = False
            else:
                dst_layer = src_layer
                preserve_fid = not explodecollections

            gfo.copy_layer(
                testfile_path,
                tmp_path,
                src_layer=src_layer,
                dst_layer=dst_layer,
                write_mode="add_layer",
                dst_crs=epsg,
                reproject=True,
                preserve_fid=preserve_fid,
                dst_dimensions=dimensions,
                explodecollections=explodecollections,
            )

            if empty:
                # Remove all rows from destination layer.
                # GDAL only supports DELETE using SQLITE dialect, not with OGRSQL.
                gfo.execute_sql(
                    tmp_path,
                    sql_stmt=f'DELETE FROM "{dst_layer}"',
                    sql_dialect="SQLITE",
                )
            elif dimensions is not None:
                if dimensions != "XYZ":
                    raise ValueError(f"unimplemented dimensions: {dimensions}")

                prepared_info = gfo.get_layerinfo(
                    tmp_path, layer=dst_layer, raise_on_nogeom=False
                )
                if prepared_info.geometrycolumn is not None:
                    gfo.update_column(
                        tmp_path,
                        name=prepared_info.geometrycolumn,
                        expression=f"CastToXYZ({prepared_info.geometrycolumn}, 5.0)",
                        layer=dst_layer,
                    )

        # Rename tmp file to prepared file
        gfo.move(tmp_path, prepared_path)

        return prepared_path

    except Exception:
        gfo.remove(prepared_path, missing_ok=True)
        raise
    finally:
        prepared_lock_path.unlink(missing_ok=True)


def set_read_only(path: Path, read_only: bool) -> None:
    """Set the file to read-only or read-write."""

    def _read_only(path: Path) -> None:
        # Set read-only
        path.chmod(S_IRUSR | S_IRGRP | S_IROTH)

    def _read_write(path: Path) -> None:
        # Set read-write
        path.chmod(S_IRWXU | S_IRWXG | S_IRWXO)

    if path.exists():
        _read_only(path) if read_only else _read_write(path)
    else:
        raise FileNotFoundError(f"File not found: {path}")

    # For some file types, extra files need to be copied
    src_info = _geofileinfo.get_geofileinfo(path)
    for s in src_info.suffixes_extrafiles:
        extra_file_path = path.parent / f"{path.stem}{s}"
        if extra_file_path.exists():
            _read_only(extra_file_path) if read_only else _read_write(extra_file_path)


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


def create_tempdir(base_dirname: str, parent_dir: Path | None = None) -> Path:
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
    check_index_type: bool | str = "equiv",
    check_column_type: bool | str = "equiv",
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
    simplify: float | None = None,
    check_geom_tolerance: float = 0.0,
    output_dir: Path | None = None,
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
    check_geom_equals : bool, default True
        If False, ignore differences between geometries.
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

    if simplify is not None:
        left.geometry = left.geometry.simplify(simplify)
        right.geometry = right.geometry.simplify(simplify)

    if promote_to_multi:
        left.geometry = _geoseries_util.harmonize_geometrytypes(
            left.geometry, force_multitype=True
        )
        right.geometry = _geoseries_util.harmonize_geometrytypes(
            right.geometry, force_multitype=True
        )

    if normalize:
        left.geometry = gpd.GeoSeries(
            shapely.normalize(left.geometry), index=left.index
        )
        right.geometry = gpd.GeoSeries(
            shapely.normalize(right.geometry),
            index=right.index,
        )

    if sort_values:
        left = geodataframe_util.sort_values(left).reset_index(drop=True)
        right = geodataframe_util.sort_values(right).reset_index(drop=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        left.to_file(output_dir / "left.geojson")
        right.to_file(output_dir / "right.geojson")

    if check_geom_tolerance > 0.0:
        # The symmetric difference should result in all empty geometries if the
        # geometries are equal. Apply a negative buffer to the geometries with half the
        # tolerance.
        symdiff = shapely.symmetric_difference(left.geometry, right.geometry)
        symdiff_tol = symdiff.buffer(-check_geom_tolerance / 2, join_style="mitre")
        symdiff_tol_diff = symdiff_tol[~symdiff_tol.is_empty]

        if not all(symdiff_tol_diff.is_empty):
            if output_dir is not None:
                # Write the differences to file
                gdf = gpd.GeoDataFrame(geometry=symdiff_tol_diff, crs=left.crs)
                gdf.to_file(output_dir / "symdiff_tol_not-empty.geojson")

            raise AssertionError(
                f"differences > {check_geom_tolerance} found in "
                f"{len(symdiff_tol_diff)} geometries: {symdiff_tol_diff=}"
            )

        right.geometry = left.geometry

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


def plot(
    geofiles: list[Path],
    output_path: Path,
    title: str | None = None,
    clean_name: bool = True,
):
    # If we are running on CI server, don't plot
    if "GITHUB_ACTIONS" in os.environ:
        return

    if mpl_figure is None or mcolors is None:
        raise ImportError(
            "matplotlib is not installed, but needed to plot geofiles. "
            "Please install matplotlib."
        )

    figure = mpl_figure.Figure(figsize=((len(geofiles) + 1) * 3, 3))
    figure.subplots(1, len(geofiles) + 1, sharex=True, sharey=True)
    if title is not None:
        figure.suptitle(title)

    colors = mcolors.TABLEAU_COLORS
    for geofile_idx, geofile in enumerate(geofiles):
        # Read the geofile
        gdf = gfo.read_file(geofile)
        if gdf is None:
            continue

        color = list(colors.keys())[geofile_idx % len(colors)]
        figure.axes[geofile_idx].set_aspect("equal")
        # add hatch to the last axis
        gdf.plot(
            ax=figure.axes[geofile_idx],
            color=colors[color],
            alpha=0.5,
        )
        gdf.plot(
            ax=figure.axes[len(geofiles)],
            color=colors[color],
            alpha=0.5,
        )
        figure.axes[geofile_idx].set_title(geofile.stem)

    if clean_name:
        # Replace all possibly invalid characters by "_"
        output_path = output_path.with_name(re.sub(r"[^\w_. -]", "_", output_path.name))

    figure.savefig(str(output_path), dpi=72)
