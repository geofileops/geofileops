# -*- coding: utf-8 -*-
"""
Module to benchmark geofileops operations.
"""

import datetime
import logging
import multiprocessing
from pathlib import Path
import sys
import tempfile
from typing import List

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import geofileops as gfo
from geofileops.util import geofileops_gpd
from geofileops.util import geofileops_sql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util import benchmark_util
from util.benchmark_util import BenchmarkResult

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

class testfile():
    AGRIPRC_2018_URL = "https://downloadagiv.blob.core.windows.net/landbouwgebruikspercelen/2018/Landbouwgebruikspercelen_LV_2018_GewVLA_Shape.zip"
    AGRIPRC_2018_NAME = "agriprc_2018.gpkg"
    AGRIPRC_2019_URL = "https://downloadagiv.blob.core.windows.net/landbouwgebruikspercelen/2019/Landbouwgebruikspercelen_LV_2019_GewVLA_Shapefile.zip"
    AGRIPRC_2019_NAME = "agriprc_2019.gpkg"
    AGRIPRC_2020_URL = "https://downloadagiv.blob.core.windows.net/landbouwgebruikspercelen/2020/Landbouwgebruikspercelen_LV_2020_GewVLA_Shapefile.zip"
    AGRIPRC_2020_NAME = "agriprc_2020.gpkg"

def geofileops_BenchResult(
        operation: str,
        secs_taken: float,
        run_details: dict) -> BenchmarkResult:
    
    return BenchmarkResult(
            version=gfo.__version__,
            package="geofileops", 
            operation=operation, 
            secs_taken=secs_taken,
            run_details=run_details)

def benchmark_buffer(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print("buffer: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf.gpkg"
    gfo.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geofileops_BenchResult(
            operation="buffer", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"buffer: ready in {secs_taken:.2f} secs")

    print("buffer with gpd: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf_gpd.gpkg"
    geofileops_gpd.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geofileops_BenchResult(
            operation="buffer_gpd", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"buffer with gpd: ready in {secs_taken:.2f} secs")
    
    print("buffer with spatialite: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf_sql.gpkg"
    geofileops_sql.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input1": input_path.name}
    results.append(geofileops_BenchResult(
            operation="buffer_spatialite", 
            secs_taken=secs_taken,
            run_details=run_details))

    print(f"buffer with spatialite: ready in {secs_taken:.2f} secs")

    return results

def benchmark_convexhull(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print("convexhull: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_convexhull.gpkg"
    gfo.convexhull(input_path, output_path, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geofileops_BenchResult(
            operation="convexhull", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"convexhull: ready in {secs_taken:.2f} secs")

    print("Start convexhull with gpd")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_convexhull_gpd.gpkg"
    geofileops_gpd.convexhull(input_path, output_path, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geofileops_BenchResult(
            operation="convexhull_gpd", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"convexhull with gpd: ready in {secs_taken:.2f} secs")
    
    print("convexhull with spatialite: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_convexhull_sql.gpkg"
    geofileops_sql.convexhull(input_path, output_path, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input1": input_path.name}
    results.append(geofileops_BenchResult(
            operation="convexhull_spatialite", 
            secs_taken=secs_taken,
            run_details=run_details))

    print(f"convexhull with spatialite: ready in {secs_taken:.2f} secs")

    return results

def benchmark_dissolve(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print("dissolve without groupby: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_nogroupby.gpkg"
    gfo.dissolve(
            input_path=input_path, 
            output_path=output_path, 
            explodecollections=True,
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input": input_path.name}
    results.append(geofileops_BenchResult(
            operation="dissolve",
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"dissolve without groupby: ready in {secs_taken:.2f} secs")
    
    print("dissolve with groupby: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_groupby.gpkg"
    geofileops_gpd.dissolve(
            input_path, 
            output_path, 
            groupby_columns=['GEWASGROEP'], 
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input": input_path.name,
            "groupby_columns": "[GEWASGROEP]"}
    results.append(geofileops_BenchResult(
            operation='dissolve_groupby', 
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"dissolve with groupby: ready in {secs_taken:.2f} secs")

    return results

def benchmark_intersect(
        input1_path: Path,
        input2_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    # Init
    if input1_path.exists() is False:
        raise Exception(f"input1_path doesn't exist: {input1_path}")
    if input2_path.exists() is False:
        raise Exception(f"input2_path doesn't exist: {input2_path}")
    results = []
    
    # Go!
    print("intersect: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    gfo.intersect(
            input1_path=input1_path, 
            input2_path=input2_path, 
            output_path=output_path,
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input1": input1_path.name,
            "input2": input2_path.name}
    results.append(geofileops_BenchResult(
            operation='intersect', 
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"intersect: ready in {secs_taken:.2f} secs")

    return results

def benchmark_simplify(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print("simplify: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_simpl.gpkg"
    gfo.simplify(input_path, output_path, tolerance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geofileops_BenchResult(
            operation="simplify", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"simplify: ready in {secs_taken:.2f} secs")
    
    print("simplify with gpd: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_simpl_gpd.gpkg"
    geofileops_gpd.simplify(input_path, output_path, tolerance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geofileops_BenchResult(
            operation="simplify_gpd", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"simplify  with gpd ready in {secs_taken:.2f} secs")

    print("simplify with spatialite: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_simpl_sql.gpkg"
    geofileops_sql.simplify(input_path, output_path, tolerance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input1": input_path.name}
    results.append(geofileops_BenchResult(
            operation="simplify_spatialite", 
            secs_taken=secs_taken,
            run_details=run_details))

    print(f"simplify with spatialite: ready in {secs_taken:.2f} secs")

    return results

def benchmark_union(
        input1_path: Path,
        input2_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    # Init
    if input1_path.exists() is False:
        raise Exception(f"input1_path doesn't exist: {input1_path}")
    if input2_path.exists() is False:
        raise Exception(f"input2_path doesn't exist: {input2_path}")
    benchmark_results = []
    
    # Go!
    print("union with spatialite: start")
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input1_path.stem}_union_{input2_path.stem}.gpkg"
    gfo.union(
            input1_path=input1_path, 
            input2_path=input2_path, 
            output_path=output_path,
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input1": input1_path.name,
            "input2": input2_path.name}
    benchmark_results.append(geofileops_BenchResult(
            operation='union',
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"union with spatialite: ready in {secs_taken:.2f} secs")

    return benchmark_results

def run(tmp_dir: Path) -> List[BenchmarkResult]:
    """
    Run the benchmarks.
    """
    # Check input params
    if tmp_dir is None:
        tmp_dir = Path(tempfile.gettempdir()) / 'geobenchmark'
        logger.info(f"tmpdir: {tmp_dir}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # First make sure the testdata is present
    input1_path = benchmark_util.download_samplefile(
            url=testfile.AGRIPRC_2018_URL,
            dst_name=testfile.AGRIPRC_2018_NAME,
            dst_dir=tmp_dir)
    input2_path = benchmark_util.download_samplefile(
            url=testfile.AGRIPRC_2019_URL,
            dst_name=testfile.AGRIPRC_2019_NAME,
            dst_dir=tmp_dir)
    
    # Now we can start benchmarking
    results = []
    #results.extend(benchmark_buffer(input1_path, tmp_dir))
    #results.extend(benchmark_convexhull(input1_path, tmp_dir))
    results.extend(benchmark_dissolve(input1_path, tmp_dir))
    """
    results.extend(benchmark_intersect(input1_path, input2_path, tmp_dir))
    results.extend(benchmark_simplify(input1_path, tmp_dir))
    results.extend(benchmark_union(input1_path, input2_path, tmp_dir))
    """

    return results