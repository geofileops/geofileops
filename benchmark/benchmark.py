# -*- coding: utf-8 -*-
"""
Module for benchmarking.
"""

import datetime
import enum
import logging
import multiprocessing
from pathlib import Path
import sys
import tempfile
from typing import List, Optional

import pandas as pd 

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import geofileops_gpd
from geofileops.util import geofileops_sql
from geofileops.util import sampledata_util
from geofileops.util.sampledata_util import SampleGeofile 

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

class Benchmark(enum.Enum):
    BUFFER = 'buffer'
    INTERSECT = 'intersect'
    UNION = 'union'
    DISSOLVE = 'dissolve'

class BenchResult:
    def __init__(self, 
            package_version: str,
            operation: str,
            tool_used: str,
            params: str,
            nb_cpu: int,
            secs_taken: float,
            secs_expected_one_cpu: float,
            input1_filename: str,
            input2_filename: Optional[str]):
        self.datetime = datetime.datetime.now()
        self.package_version = package_version
        self.operation = operation
        self.tool_used = tool_used
        self.params = params
        self.nb_cpu = nb_cpu
        self.secs_taken = secs_taken
        self.secs_expected_one_cpu = secs_expected_one_cpu
        self.input1_filename = input1_filename
        self.input2_filename = input2_filename
        
    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"

def get_geofileops_version():
    version_path = Path(__file__).resolve().parent.parent / 'version.txt'
    with open(version_path, mode='r') as file:
        return file.readline()

def benchmark_buffer(tmpdir: Path) -> List[BenchResult]:
    
    bench_results = []
    input_path = SampleGeofile.POLYGON_AGRI_PARCEL_2019.custompath(tmpdir)
    
    print('Start buffer with geopandas')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf_gpd.gpkg"
    geofileops_gpd.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=get_geofileops_version(),
            operation='buffer', 
            tool_used='geopandas', 
            params='',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=1000,
            input1_filename=input_path.name, 
            input2_filename=None))
    print(f"Buffer with geopandas ready in {secs_taken:.2f} secs")
    
    print('Start buffer with sql')

    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf_sql.gpkg"
    geofileops_sql.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=get_geofileops_version(),
            operation='buffer', 
            tool_used='sql', 
            params='',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=1600,
            input1_filename=input_path.name, 
            input2_filename=None))

    print(f"Buffer with sql ready in {secs_taken:.2f} secs")

    return bench_results

def benchmark_dissolve(tmpdir: Path) -> List[BenchResult]:
    
    bench_results = []
    input_path = SampleGeofile.POLYGON_AGRI_PARCEL_2019.custompath(tmpdir)
    
    print('Dissolve without groupby: start')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_nogroupby.gpkg"
    geofileops_gpd.dissolve(
            input_path=input_path, 
            output_path=output_path, 
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=get_geofileops_version(),
            operation='dissolve', 
            tool_used='geopandas', 
            params='',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=2000,
            input1_filename=input_path.name, 
            input2_filename=None))
    print(f"Dissolve without groupby ready in {secs_taken:.2f} secs")
    
    print('Dissolve with groupby: start')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_groupby.gpkg"
    geofileops_gpd.dissolve(
            input_path, 
            output_path, 
            groupby_columns=['GEWASGROEP'], 
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=get_geofileops_version(),
            operation='dissolve', 
            tool_used='geopandas', 
            params='groupby_columns=[GEWASGROEP]',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=3000,
            input1_filename=input_path.name, 
            input2_filename=None))
    print(f"Dissolve with groupby ready in {secs_taken:.2f} secs")

    print('Dissolve with groupby + no explode: start')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_groupby_noexplode.gpkg"
    geofileops_gpd.dissolve(
            input_path, 
            output_path, 
            groupby_columns=['GEWASGROEP'], 
            explodecollections=False, 
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=get_geofileops_version(),
            operation='dissolve', 
            tool_used='geopandas', 
            params='groupby_columns=[GEWASGROEP];explodecollection=False',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=3000,
            input1_filename=input_path.name, 
            input2_filename=None))
    print(f"Dissolve with groupby + no explode ready in {secs_taken:.2f} secs")

    return bench_results

def benchmark_intersect(tmpdir: Path) -> List[BenchResult]:
    # Init
    bench_results = []
    input1_path = SampleGeofile.POLYGON_AGRI_PARCEL_2019.custompath(tmpdir)
    input2_path = SampleGeofile.POLYGON_AGRI_PARCEL_2018.custompath(tmpdir)
    
    print('Start Intersect with sql')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    geofileops_sql.intersect(
            input1_path=input1_path, 
            input2_path=input2_path, 
            output_path=output_path,
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=get_geofileops_version(),
            operation='intersect', 
            tool_used='sql', 
            params='',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=1700,
            input1_filename=input1_path.name, 
            input2_filename=input2_path.name))
    print(f"Intersect with sql ready in {secs_taken:.2f} secs")

    return bench_results

def benchmark_union(tmpdir: Path) -> List[BenchResult]:
    # Init
    bench_results = []
    input1_path = SampleGeofile.POLYGON_AGRI_PARCEL_2019.custompath(tmpdir)
    input2_path = SampleGeofile.POLYGON_AGRI_PARCEL_2018.custompath(tmpdir)
    
    print('Start Union with sql')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input1_path.stem}_union_{input2_path.stem}.gpkg"
    geofileops_sql.union(
            input1_path=input1_path, 
            input2_path=input2_path, 
            output_path=output_path,
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=get_geofileops_version(),
            operation='union', 
            tool_used='sql', 
            params='',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=4000,
            input1_filename=input1_path.name, 
            input2_filename=input2_path.name))
    
    print(f"Union with sql ready in {secs_taken:.2f} secs")

    return bench_results

def benchmark(
        benchmarks_to_run: List[Benchmark],
        tmpdir: Path = None):
    """
    Run the benchmarks specified.

    Args:
        benchmarks_to_run (List[Benchmark]): [description]
    """
    # Check input params
    if tmpdir is None:
        tmpdir = Path(tempfile.gettempdir()) / 'geofileops_benchmark'
        logger.info(f"tmpdir: {tmpdir}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    
    # First make sure the testdata is present
    sampledata_util.download_samplefile(SampleGeofile.POLYGON_AGRI_PARCEL_2018, tmpdir)
    sampledata_util.download_samplefile(SampleGeofile.POLYGON_AGRI_PARCEL_2019, tmpdir)
    
    # Now we can start benchmarking
    results = []
    if Benchmark.BUFFER in benchmarks_to_run:
        results.extend(benchmark_buffer(tmpdir))
    if Benchmark.INTERSECT in benchmarks_to_run:
        results.extend(benchmark_intersect(tmpdir))
    if Benchmark.UNION in benchmarks_to_run:
        results.extend(benchmark_union(tmpdir))
    if Benchmark.DISSOLVE in benchmarks_to_run:
        results.extend(benchmark_dissolve(tmpdir))
    
    # Check and print results
    for result in results:
        if result.secs_taken > ((result.secs_expected_one_cpu/result.nb_cpu) * 1.5):
            print(f"ERROR: {result}") 
        else:
            print(str(result))
    
    # Write results to csv file
    results_path = Path(__file__).resolve().parent / 'benchmark_results.csv'
    results_dictlist = [vars(result) for result in results]
    results_df = pd.DataFrame(results_dictlist)

    # If output file doesn't exist yet, create new, otherwise append...
    if not results_path.exists():
        results_df.to_csv(results_path, index=False)
    else:
        results_df.to_csv(results_path, index=False, mode='a', header=False)

if __name__ == '__main__':
    # Init logging
    logging.basicConfig(
            format="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s", 
            datefmt="%H:%M:%S", level=logging.INFO)

    #Go!
    tmpdir = None
    #tmpdir = Path(r"C:\Temp") / 'geofileops_benchmark'
    benchmark(
            benchmarks_to_run=[
                Benchmark.BUFFER,
                Benchmark.UNION,
                Benchmark.INTERSECT,
                Benchmark.DISSOLVE,
            ],
            tmpdir=tmpdir)
