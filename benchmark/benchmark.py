# -*- coding: utf-8 -*-
"""
Module for benchmarking.
"""

import datetime
import enum
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import List, Optional
import urllib.request
import zipfile

import pandas as pd 

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import geofileops_gpd
from geofileops.util import geofileops_sql
from geofileops import geofileops

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

def _get_testdata_dir(tmpdir: Path) -> Path:
    testdata_dir = tmpdir / 'data'
    return testdata_dir

def _get_testdata_aiv(testdata_path: Path):

    # If the testdata file already exists... just return
    if testdata_path.exists():
        return
    elif testdata_path.name.lower() == 'Lbgbrprc19.gpkg'.lower():
        url = r"https://downloadagiv.blob.core.windows.net/landbouwgebruikspercelen/2019/Landbouwgebruikspercelen_LV_2019_GewVLA_Shapefile.zip"
    elif testdata_path.name.lower() == 'Lbgbrprc18.gpkg'.lower():
        url = r"https://downloadagiv.blob.core.windows.net/landbouwgebruikspercelen/2018/Landbouwgebruikspercelen_LV_2018_GewVLA_Shape.zip"
    else:
        raise Exception(f"Unknown testdata file: {testdata_path}")

    # Download zip file if needed...  
    zip_path = testdata_path.parent / f"{testdata_path.stem}.zip"
    unzippedzip_dir = testdata_path.parent / zip_path.stem
    if not zip_path.exists() and not unzippedzip_dir.exists():
        # Download beschmark file
        print(f"Download test data {testdata_path}")
        urllib.request.urlretrieve(str(url), zip_path)
    
    # Unzip zip file if needed... 
    if not unzippedzip_dir.exists():
        # Unzip file
        print('Unzip test data')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzippedzip_dir)
            
    # Convert shapefile to geopackage
    shp_dir = unzippedzip_dir / 'Shapefile'
    shp_paths = list(shp_dir.glob('Lbgbrprc*.shp'))
    if len(shp_paths) != 1:
        raise Exception(f"Should find 1 shapefile, found {len(shp_paths)}")

    print('Make shapefile valid to gpkg')
    geofileops.makevalid(shp_paths[0], testdata_path)
    
    # Cleanup
    if zip_path.exists():
        zip_path.unlink()
    if unzippedzip_dir.exists():
        shutil.rmtree(unzippedzip_dir)

def _get_package_version():
    version_path = Path(__file__).resolve().parent.parent / 'version.txt'
    with open(version_path, mode='r') as file:
        return file.readline()

def benchmark_buffer(tmpdir: Path) -> List[BenchResult]:
    
    bench_results = []
    input_path = _get_testdata_dir(tmpdir) / 'Lbgbrprc19.gpkg'
    
    print('Start buffer with geopandas')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_buf_gpd.gpkg"
    geofileops_gpd.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=_get_package_version(),
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
            package_version=_get_package_version(),
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
    input_path = _get_testdata_dir(tmpdir) / 'Lbgbrprc19.gpkg'
    
    print('Dissolve without groupby: start')
    start_time = datetime.datetime.now()
    output_path = tmpdir / f"{input_path.stem}_diss_nogroupby.gpkg"
    geofileops_gpd.dissolve(
            input_path, 
            output_path, 
            force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            package_version=_get_package_version(),
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
            package_version=_get_package_version(),
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
            package_version=_get_package_version(),
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
    input1_path = _get_testdata_dir(tmpdir) / 'Lbgbrprc19.gpkg'
    input2_path = _get_testdata_dir(tmpdir) / 'Lbgbrprc18.gpkg'
    
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
            package_version=_get_package_version(),
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
    input1_path = _get_testdata_dir(tmpdir) / 'Lbgbrprc19.gpkg'
    input2_path = _get_testdata_dir(tmpdir) / 'Lbgbrprc18.gpkg'
    
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
            package_version=_get_package_version(),
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
    
    # First make sure the testdata is present
    testdata_dir = _get_testdata_dir(tmpdir=tmpdir)
    testdata_dir.mkdir(parents=True, exist_ok=True)
    prc2019_path = testdata_dir / 'Lbgbrprc19.gpkg'
    _get_testdata_aiv(prc2019_path)
    prc2018_path = testdata_dir / 'Lbgbrprc18.gpkg'
    _get_testdata_aiv(prc2018_path)

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
    #tmpdir = Path(r"X:\PerPersoon\PIEROG\Temp") / 'geofileops_benchmark'
    benchmark(
            benchmarks_to_run=[
                #Benchmark.BUFFER,
                Benchmark.UNION,
                Benchmark.INTERSECT,
                #Benchmark.DISSOLVE,
            ],
            tmpdir=tmpdir)
