"""
Module to benchmark geofileops operations.
"""

from datetime import datetime
import logging
import multiprocessing
import inspect
from pathlib import Path

import geopandas as gpd
import shapely

from benchmark.benchmarker import RunResult
from benchmark.benchmarks import testdata
import geofileops as gfo
from geofileops.util import _geoops_sql
from geofileops.util import _geoops_gpd

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)
nb_parallel = min(multiprocessing.cpu_count(), 12)

################################################################################
# The real work
################################################################################


def _get_package() -> str:
    return "geofileops"


def _get_version() -> str:
    return gfo.__version__


def buffer(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf.gpkg"
    gfo.buffer(input_path, output_path, distance=1, nb_parallel=nb_parallel, force=True)
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="buffer",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr="buffer on agri parcel layer BEFL (~500.000 polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def buffer_spatialite(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_spatialite.gpkg"
    _geoops_sql.buffer(
        input_path, output_path, distance=1, nb_parallel=nb_parallel, force=True
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="buffer_spatialite",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr="buffer on agri parcel layer BEFL (~500.000 polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def buffer_gridsize_spatialite(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_grid01_spatialite.gpkg"
    _geoops_sql.buffer(
        input_path,
        output_path,
        distance=1,
        gridsize=0.1,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="buffer_gridsize_spatialite",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=(
            "buffer with gridsize 0.1 on agri parcel layer BEFL (~500.000 polygons)"
        ),
        run_details={"nb_cpu": multiprocessing.cpu_count()},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def buffer_gpd(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_gpd.gpkg"
    _geoops_gpd.buffer(
        input_path, output_path, distance=1, nb_parallel=nb_parallel, force=True
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="buffer_gpd",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr="buffer on agri parcel layer BEFL (~500.000 polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def dissolve_nogroupby(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_diss_nogroupby.gpkg"
    gfo.dissolve(
        input_path=input_path,
        output_path=output_path,
        explodecollections=True,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="dissolve",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr="dissolve on agri parcels BEFL (~500.000 polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def dissolve_groupby(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_diss_groupby.gpkg"
    gfo.dissolve(
        input_path,
        output_path,
        groupby_columns=["GEWASGROEP"],
        explodecollections=True,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="dissolve_groupby",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=(
            "dissolve on agri parcels BEFL (~500.000 polygons), groupby=[GEWASGROEP]"
        ),
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def clip(tmp_dir: Path) -> RunResult:
    # Init
    input1_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    input2_path = testdata.TestFile.AGRIPRC_2019.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input1_path.stem}_clip_{input2_path.stem}.gpkg"
    gfo.clip(
        input_path=input1_path,
        clip_path=input2_path,
        output_path=output_path,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package="geofileops",
        package_version=gfo.__version__,
        operation="clip",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr="clip between 2 agri parcel layers BEFL (2*~500.000 polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    # output_path.unlink()
    return result


def intersection(tmp_dir: Path) -> RunResult:
    # Init
    input1_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    input2_path = testdata.TestFile.AGRIPRC_2019.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    gfo.intersection(
        input1_path=input1_path,
        input2_path=input2_path,
        output_path=output_path,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="intersection",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=(
            "intersection between 2 agri parcel layers BEFL (2*~500.000 polygons)"
        ),
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def intersection_complexpoly(tmp_dir: Path) -> RunResult:
    # Init
    function_name = inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
    input1_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    # Prepare a complex polygon to test with
    poly_complex = _create_complex_poly(
        xmin=30000.123,
        ymin=170000.123,
        width=10000,
        height=10000,
        line_distance=500,
        max_segment_length=100,
    )
    print(f"num_coordinates: {shapely.get_num_coordinates(poly_complex)}")
    input2_path = tmp_dir / "complex.gpkg"
    complex_gdf = gpd.GeoDataFrame(geometry=[poly_complex], crs="epsg:31370")
    complex_gdf.to_file(input2_path, engine="pyogrio")

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    gfo.intersection(
        input1_path=input1_path,
        input2_path=input2_path,
        output_path=output_path,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation=function_name,
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=(
            f"{function_name} between complex + 1 agri parcel layers BEFL"
        ),
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def intersection_gridsize(tmp_dir: Path) -> RunResult:
    # Init
    input1_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    input2_path = testdata.TestFile.AGRIPRC_2019.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input1_path.stem}_inters_grid01_{input2_path.stem}.gpkg"
    gfo.intersection(
        input1_path=input1_path,
        input2_path=input2_path,
        output_path=output_path,
        gridsize=0.001,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="intersection_gridsize",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=(
            "intersection with gridsize 0.001 between 2 agri parcel layers BEFL "
            "(2*~500k polygons)"
        ),
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def join_by_location_intersects(tmp_dir: Path) -> RunResult:
    # Init-
    input1_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    input2_path = testdata.TestFile.AGRIPRC_2019.get_file(tmp_dir)

    '''
    input2_all_path = testdata.TestFile.COMMUNES.get_file(tmp_dir)
    input2_path = input2_all_path.parent / f"{input2_all_path.stem}_filtered.gpkg"
    sql_stmt = f"""
                SELECT *
                    FROM "{{input_layer}}" layer
                    WHERE """
    gfo.select(
            input_path=input2_all_path,
            output_path=input2_path,
            sql_stmt=sql_stmt)
    '''

    # Go!
    start_time = datetime.now()
    output_path = (
        tmp_dir
        / f"{input1_path.stem}_join_inters_{input2_path.stem}_{_get_package()}.gpkg"
    )
    gfo.join_by_location(
        input1_path=input1_path,
        input2_path=input2_path,
        output_path=output_path,
        spatial_relations_query="intersects is True",
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="join_by_location_intersects",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=(
            "join_by_location_intersects between 2 agri parcel layers BEFL "
            "(2*~500.000 polygons)"
        ),
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    logger.info(f"nb features in result: {gfo.get_layerinfo(output_path).featurecount}")
    output_path.unlink()
    return result


def makevalid_gridsize_gpd(tmp_dir: Path) -> RunResult:
    # Init
    function_name = inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_{function_name}.gpkg"
    gfo.makevalid(
        input_path, output_path, gridsize=0.001, nb_parallel=nb_parallel, force=True
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation=function_name,
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=f"{function_name} on agri parcel layer BEFL (~500k polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def makevalid_gridsize_spatialite(tmp_dir: Path) -> RunResult:
    # Init
    function_name = inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_{function_name}.gpkg"
    _geoops_sql.makevalid(
        input_path, output_path, gridsize=0.001, nb_parallel=nb_parallel, force=True
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation=function_name,
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=f"{function_name} on agri parcel layer BEFL (~500k polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def makevalid_gpd(tmp_dir: Path) -> RunResult:
    # Init
    function_name = inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_{function_name}.gpkg"
    gfo.makevalid(input_path, output_path, nb_parallel=nb_parallel, force=True)
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation=function_name,
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=f"{function_name} on agri parcel layer BEFL (~500k polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def makevalid_spatialite(tmp_dir: Path) -> RunResult:
    # Init
    function_name = inspect.currentframe().f_code.co_name  # type: ignore[union-attr]
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_{function_name}.gpkg"
    _geoops_sql.makevalid(input_path, output_path, nb_parallel=nb_parallel, force=True)
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation=function_name,
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr=f"{function_name} on agri parcel layer BEFL (~500k polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def union(tmp_dir: Path) -> RunResult:
    # Init
    input1_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    input2_path = testdata.TestFile.AGRIPRC_2019.get_file(tmp_dir)

    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    gfo.union(
        input1_path=input1_path,
        input2_path=input2_path,
        output_path=output_path,
        nb_parallel=nb_parallel,
        force=True,
    )
    result = RunResult(
        package=_get_package(),
        package_version=_get_version(),
        operation="union",
        secs_taken=(datetime.now() - start_time).total_seconds(),
        operation_descr="union between 2 agri parcel layers BEFL (2*~500.000 polygons)",
        run_details={"nb_cpu": nb_parallel},
    )

    # Cleanup and return
    output_path.unlink()
    return result


def _create_complex_poly(
    xmin: float,
    ymin: float,
    width: int,
    height: int,
    line_distance: int,
    max_segment_length: int,
) -> shapely.Polygon:
    """Create complex polygon of a ~grid-shape the size specified."""
    lines = []

    # Vertical lines
    for x_offset in range(0, 0 + width, line_distance):
        lines.append(
            shapely.LineString(
                [(xmin + x_offset, ymin), (xmin + x_offset, ymin + height)]
            )
        )

    # Horizontal lines
    for y_offset in range(0, 0 + height, line_distance):
        lines.append(
            shapely.LineString(
                [(xmin, ymin + y_offset), (xmin + width, ymin + y_offset)]
            )
        )

    poly_complex = shapely.unary_union(shapely.MultiLineString(lines).buffer(2))
    poly_complex = shapely.segmentize(
        poly_complex, max_segment_length=max_segment_length
    )
    assert len(shapely.get_parts(poly_complex)) == 1

    return poly_complex
