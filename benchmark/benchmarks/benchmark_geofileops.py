# -*- coding: utf-8 -*-
"""
Module to benchmark geofileops operations.
"""

from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
from typing import List

import geofileops as gfo
from geofileops.util import _geoops_gpd
from geofileops.util import _geoops_sql

from benchmark.benchmarker import RunResult
from . import testdata

################################################################################
# Some inits
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def geofileops_RunResult(
        operation: str,
        operation_descr: str,
        secs_taken: float,
        run_details: dict) -> RunResult:
    
    return RunResult(
            package="geofileops", 
            package_version=gfo.__version__,
            operation=operation,
            operation_descr=operation_descr, 
            secs_taken=secs_taken,
            run_details=run_details)

def benchmark_buffer(
        input_path: Path,
        tmpdir: Path) -> List[RunResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print("buffer: start")
    start_time = datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf.gpkg"
    gfo.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.now()-start_time).total_seconds()
    results.append(geofileops_RunResult(
            operation="buffer", 
            operation_descr="buffer agri parcels BEFL (~500.000 polygons)",
            secs_taken=secs_taken,
            run_details={"nb_cpu": multiprocessing.cpu_count()}))
    print(f"buffer: ready in {secs_taken:.2f} secs")

    print("buffer with gpd: start")
    start_time = datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf_gpd.gpkg"
    _geoops_gpd.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.now()-start_time).total_seconds()
    results.append(geofileops_RunResult(
            operation="buffer_gpd", 
            operation_descr="buffer agri parcels BEFL (~500.000 polygons), using gpd",
            secs_taken=secs_taken,
            run_details={"nb_cpu": multiprocessing.cpu_count()}))
    print(f"buffer with gpd: ready in {secs_taken:.2f} secs")
    
    print("buffer with spatialite: start")
    start_time = datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf_sql.gpkg"
    _geoops_sql.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.now()-start_time).total_seconds()
    results.append(geofileops_RunResult(
            operation="buffer_spatialite", 
            operation_descr="buffer agri parcels BEFL (~500.000 polygons), using spatialite",
            secs_taken=secs_taken,
            run_details={"nb_cpu": multiprocessing.cpu_count()}))

    print(f"buffer with spatialite: ready in {secs_taken:.2f} secs")

    return results

def benchmark_dissolve(
        input_path: Path,
        tmpdir: Path) -> List[RunResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print("dissolve without groupby: start")
    start_time = datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_nogroupby.gpkg"
    gfo.dissolve(
            input_path=input_path, 
            output_path=output_path, 
            explodecollections=True,
            force=True)
    secs_taken = (datetime.now()-start_time).total_seconds()
    results.append(geofileops_RunResult(
            operation="dissolve",
            secs_taken=secs_taken,
            operation_descr="dissolve on agri parcels BEFL (~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()}))
    print(f"dissolve without groupby: ready in {secs_taken:.2f} secs")
    
    print("dissolve with groupby: start")
    start_time = datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_groupby.gpkg"
    _geoops_gpd.dissolve(
            input_path, 
            output_path, 
            groupby_columns=['GEWASGROEP'], 
            force=True)
    secs_taken = (datetime.now()-start_time).total_seconds()
    results.append(geofileops_RunResult(
            operation='dissolve_groupby', 
            secs_taken=secs_taken,
            operation_descr="dissolve on agri parcels BEFL (~500.000 polygons), groupby=[GEWASGROEP]",
            run_details={"nb_cpu": multiprocessing.cpu_count()}))
    print(f"dissolve with groupby: ready in {secs_taken:.2f} secs")

    return results

def benchmark_intersect(
        input1_path: Path,
        input2_path: Path,
        tmpdir: Path) -> List[RunResult]:
    # Init
    if input1_path.exists() is False:
        raise Exception(f"input1_path doesn't exist: {input1_path}")
    if input2_path.exists() is False:
        raise Exception(f"input2_path doesn't exist: {input2_path}")
    results = []
    
    # Go!
    print("intersect: start")
    start_time = datetime.now()
    output_path = tmpdir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    gfo.intersect(
            input1_path=input1_path, 
            input2_path=input2_path, 
            output_path=output_path,
            force=True)
    secs_taken = (datetime.now()-start_time).total_seconds()
    results.append(geofileops_RunResult(
            operation='intersect', 
            secs_taken=secs_taken,
            operation_descr="intersect between 2 agri parcel layers BEFL (2*~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()}))
    print(f"intersect: ready in {secs_taken:.2f} secs")

    return results

def benchmark_union(
        input1_path: Path,
        input2_path: Path,
        tmpdir: Path) -> List[RunResult]:
    # Init
    if input1_path.exists() is False:
        raise Exception(f"input1_path doesn't exist: {input1_path}")
    if input2_path.exists() is False:
        raise Exception(f"input2_path doesn't exist: {input2_path}")
    benchmark_results = []
    
    # Go!
    print("union with spatialite: start")
    start_time = datetime.now()
    output_path = tmpdir / f"{input1_path.stem}_union_{input2_path.stem}.gpkg"
    gfo.union(
            input1_path=input1_path, 
            input2_path=input2_path, 
            output_path=output_path,
            force=True)
    secs_taken = (datetime.now()-start_time).total_seconds()
    benchmark_results.append(geofileops_RunResult(
            operation='union',
            secs_taken=secs_taken,
            operation_descr="union between 2 agri parcel layers BEFL (2*~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()}))
    print(f"union with spatialite: ready in {secs_taken:.2f} secs")

    return benchmark_results

def run(tmp_dir: Path) -> List[RunResult]:
    """
    Run the benchmarks.
    """
    # Get the test data
    agriprc2018_path, agriprc2019_path = testdata.get_testdata(tmp_dir)

    # Now we can start benchmarking
    results = []
    results.extend(benchmark_buffer(agriprc2018_path, tmp_dir))
    results.extend(benchmark_dissolve(agriprc2018_path, tmp_dir))
    results.extend(benchmark_intersect(agriprc2018_path, agriprc2019_path, tmp_dir))
    results.extend(benchmark_union(agriprc2018_path, agriprc2019_path, tmp_dir))
    
    return results