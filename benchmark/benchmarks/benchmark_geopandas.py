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

import geopandas as gpd

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import geofileops as gfo

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

def geopandas_BenchResult(
        operation: str,
        secs_taken: float,
        run_details: dict) -> BenchmarkResult:
    
    return BenchmarkResult(
            version=gpd.__version__,
            package="geopandas", 
            operation=operation, 
            secs_taken=secs_taken,
            run_details=run_details)

def benchmark_buffer(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    ### Init ###
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    ### Go! ###
    print("buffer start")

    # Read input file
    start_time = datetime.datetime.now()
    # This read actually used pyogrio, so is not really geopandas
    gdf = gfo.read_file(input_path)
    
    secs_read = (datetime.datetime.now()-start_time).total_seconds()
    print(f"time for read: {secs_read}")
    
    # Buffer
    start_time_buffer = datetime.datetime.now()
    gdf.geometry = gdf.geometry.buffer(distance=1, resolution=5)
    secs_onlybuffer = (datetime.datetime.now()-start_time_buffer).total_seconds()
    print(f"time for buffer: {secs_onlybuffer}")
    results.append(geopandas_BenchResult(
            operation="buffer_noIO", 
            secs_taken=secs_onlybuffer,
            run_details={"input1": input_path.name}))
    
    # Write to output file
    start_time_write = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_geopandas_buf.gpkg"
    # This read actually used pyogrio, so is not really geopandas
    gfo.to_file(gdf, output_path)
    #output_gdf.to_file(output_path, layer=output_path.stem, driver="GPKG")
    secs_write = (datetime.datetime.now()-start_time_write).total_seconds()
    print(f"write took {secs_write}")
    
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geopandas_BenchResult(
            operation="buffer", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"Buffer ready in {secs_taken:.2f} secs")

    return results

'''
def benchmark_convexhull(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print('Start convexhull')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_convexhull.gpkg"
    gfo.convexhull(input_path, output_path, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geopandas_BenchResult(
            operation="convexhull", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"convexhull ready in {secs_taken:.2f} secs")

    return results

def benchmark_dissolve(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print('Dissolve without groupby: start')
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
    results.append(geopandas_BenchResult(
            operation="dissolve",
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"Dissolve without groupby ready in {secs_taken:.2f} secs")
    
    print('Dissolve with groupby: start')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_groupby.gpkg"
    gfo.dissolve(
            input_path, 
            output_path, 
            groupby_columns=['GEWASGROEP'],
            explodecollections=True, 
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    run_details = {
            "nb_cpu_used": multiprocessing.cpu_count(),
            "input": input_path.name,
            "groupby_columns": "[GEWASGROEP]"}
    results.append(geopandas_BenchResult(
            operation='dissolve_groupby', 
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"Dissolve with groupby ready in {secs_taken:.2f} secs")

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
    print('Start Intersect')
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
    results.append(geopandas_BenchResult(
            operation='intersect', 
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"Intersect ready in {secs_taken:.2f} secs")

    return results

def benchmark_simplify(
        input_path: Path,
        tmpdir: Path) -> List[BenchmarkResult]:
    
    # Init
    if input_path.exists() is False:
        raise Exception(f"input_path doesn't exist: {input_path}")
    results = []
    
    # Go!
    print('Start simplify')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_simpl.gpkg"
    gfo.simplify(input_path, output_path, tolerance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    results.append(geopandas_BenchResult(
            operation="simplify", 
            secs_taken=secs_taken,
            run_details={"input1": input_path.name}))
    print(f"simplify ready in {secs_taken:.2f} secs")
    
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
    print('Start Union with sql')
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
    benchmark_results.append(geopandas_BenchResult(
            operation='union',
            secs_taken=secs_taken,
            run_details=run_details))
    print(f"Union ready in {secs_taken:.2f} secs")

    return benchmark_results
'''

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
    """
    results.extend(benchmark_buffer(input1_path, tmp_dir))
    results.extend(benchmark_convexhull(input1_path, tmp_dir))
    results.extend(benchmark_dissolve(input1_path, tmp_dir))
    results.extend(benchmark_intersect(input1_path, input2_path, tmp_dir))
    results.extend(benchmark_simplify(input1_path, tmp_dir))
    results.extend(benchmark_union(input1_path, input2_path, tmp_dir))
    """

    return results