# -*- coding: utf-8 -*-
"""
Module to benchmark geopandas operations.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd

from benchmark.benchmarker import RunResult
from . import testdata

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def geopandas_RunResult(
        operation: str,
        operation_descr: str,
        secs_taken: float,
        run_details: Optional[dict] = None) -> RunResult:
    
    return RunResult(
            package="geopandas", 
            package_version=gpd.__version__,
            operation=operation, 
            operation_descr=operation_descr,
            secs_taken=secs_taken,
            run_details=run_details)

def benchmark_buffer(
        input_path: Path,
        tmpdir: Path) -> List[RunResult]:
    
    ### Init ###
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")

    ### Go! ###
    # Read input file
    print(f"buffer start")
    start_time = datetime.now()
    gdf = gpd.read_file(input_path)
    print(f"time for read: {(datetime.now()-start_time).total_seconds()}")
    
    # Buffer
    start_time_buffer = datetime.now()
    gdf.geometry = gdf.geometry.buffer(distance=1, resolution=5)
    print(f"time for buffer: {(datetime.now()-start_time_buffer).total_seconds()}")
    
    # Write to output file
    start_time_write = datetime.now()
    output_path = tmpdir / f"{input_path.stem}_geopandas_buf.gpkg"
    # This read actually used pyogrio, so is not really geopandas
    gdf.to_file(output_path, layer=output_path.stem, driver="GPKG")
    print(f"write took {(datetime.now()-start_time_write).total_seconds()}")    
    result = geopandas_RunResult(
            operation="buffer", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="buffer agri parcels BEFL (~500.000 polygons)")
    print(f"{result.operation} took {result.secs_taken} secs")

    # Cleanup
    output_path.unlink()

    return [result]

def benchmark_dissolve(
        input_path: Path,
        tmpdir: Path) -> List[RunResult]:
    
    ### Init ###
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")

    ### Go! ###
    # Read input file
    print(f"dissolve start")
    start_time = datetime.now()
    gdf = gpd.read_file(input_path)
    print(f"time for read: {(datetime.now()-start_time).total_seconds()}")
    
    # dissolve
    start_time_dissolve = datetime.now()
    gdf = gdf.dissolve()
    print(f"time for dissolve: {(datetime.now()-start_time_dissolve).total_seconds()}")
    
    # Write to output file
    start_time_write = datetime.now()
    output_path = tmpdir / f"{input_path.stem}_geopandas_diss.gpkg"
    gdf.to_file(output_path, layer=output_path.stem, driver="GPKG")
    print(f"write took {(datetime.now()-start_time_write).total_seconds()}")
    result = geopandas_RunResult(
            operation="dissolve", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="dissolve agri parcels BEFL (~500.000 polygons)")
    print(f"{result.operation} took {result.secs_taken} secs")

    # Cleanup
    output_path.unlink()

    return [result]

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

    return results