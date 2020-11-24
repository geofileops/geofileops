import datetime
import multiprocessing
from pathlib import Path
import shutil
import sys
import tempfile
from typing import List, Optional
import urllib.request
import zipfile

import pandas as pd 

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))
from geofileops.util import geofileops_gpd
from geofileops.util import geofileops_ogr
from geofileops import gfo_geometry

class BenchResult:
    def __init__(self, 
            operation: str,
            tool_used: str,
            params: str,
            nb_cpu: int,
            secs_taken: float,
            secs_expected_one_cpu: float,
            input1_filename: str,
            input2_filename: Optional[str]):
        self.datetime = datetime.datetime.now()
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

class GdalBin():
    def __init__(self, gdal_installation: str, gdal_bin_path: str = None):
        self.gdal_installation = gdal_installation
        if gdal_installation == 'gdal_bin':
            if gdal_bin_path is None:
                self.gdal_bin_path = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
            else:
                self.gdal_bin_path = gdal_bin_path

    def __enter__(self):
        if self.gdal_installation == 'gdal_bin':
            import os
            os.environ['GDAL_BIN'] = self.gdal_bin_path

    def __exit__(self, type, value, traceback):
        #Exception handling here
        import os
        if os.environ.get('GDAL_BIN') is not None:
            del os.environ['GDAL_BIN']

def get_testdata_dir() -> Path:
    #return Path(__file__).resolve().parent / 'data'
    testdata_dir = Path(tempfile.gettempdir()) / 'geofileops_benchmark_data'
    return testdata_dir

def get_testdata_aiv(testdata_path: Path):

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
    zip_path = get_testdata_dir() / f"{testdata_path.stem}.zip"
    unzippedzip_dir = get_testdata_dir() / zip_path.stem
    if not zip_path.exists() and not unzippedzip_dir.exists():
        # Download beschmark file
        print('Download test data')
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
    gfo_geometry.makevalid(shp_paths[0], testdata_path)
    
    # Cleanup
    if zip_path.exists():
        zip_path.unlink()
    if unzippedzip_dir.exists():
        shutil.rmtree(unzippedzip_dir)

def benchmark_buffer(tmp_dir: Path) -> List[BenchResult]:
    
    bench_results = []
    input_path = get_testdata_dir() / 'Lbgbrprc19.gpkg'
    
    print('Start buffer with geopandas')
    start_time = datetime.datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_gpd.gpkg"
    geofileops_gpd.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            operation='buffer', 
            tool_used='geopandas', 
            params='',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=1000,
            input1_filename=input_path.name, 
            input2_filename=None))
    print(f"Buffer with geopandas ready in {secs_taken:.2f} secs")
    
    print('Start buffer with ogr')

    start_time = datetime.datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_ogr.gpkg"
    geofileops_ogr.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(BenchResult(
            operation='buffer', 
            tool_used='ogr', 
            params='',
            nb_cpu=multiprocessing.cpu_count(),
            secs_taken=secs_taken,
            secs_expected_one_cpu=1600,
            input1_filename=input_path.name, 
            input2_filename=None))

    print(f"Buffer with ogr ready in {secs_taken:.2f} secs")

    return bench_results

def benchmark_intersect(tmp_dir: Path) -> List[BenchResult]:
    # Init
    bench_results = []
    input1_path = get_testdata_dir() / 'Lbgbrprc19.gpkg'
    input2_path = get_testdata_dir() / 'Lbgbrprc18.gpkg'
    
    with GdalBin('gdal_bin'):
        print('Start Intersect with ogr')
        start_time = datetime.datetime.now()
        output_path = tmp_dir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
        geofileops_ogr.intersect(
                input1_path=input1_path, 
                input2_path=input2_path, 
                output_path=output_path,
                force=True)
        secs_taken = (datetime.datetime.now()-start_time).total_seconds()
        bench_results.append(BenchResult(
                operation='intersect', 
                tool_used='ogr', 
                params='',
                nb_cpu=multiprocessing.cpu_count(),
                secs_taken=secs_taken,
                secs_expected_one_cpu=1700,
                input1_filename=input1_path.name, 
                input2_filename=input2_path.name))
        print(f"Intersect with ogr ready in {secs_taken:.2f} secs")

    return bench_results

def benchmark_union(tmp_dir: Path) -> List[BenchResult]:
    # Init
    bench_results = []
    input1_path = get_testdata_dir() / 'Lbgbrprc19.gpkg'
    input2_path = get_testdata_dir() / 'Lbgbrprc18.gpkg'
    
    with GdalBin('gdal_bin'):
        print('Start Union with ogr')
        start_time = datetime.datetime.now()
        output_path = tmp_dir / f"{input1_path.stem}_union_{input2_path.stem}.gpkg"
        geofileops_ogr.union(
                input1_path=input1_path, 
                input2_path=input2_path, 
                output_path=output_path,
                force=True)
        secs_taken = (datetime.datetime.now()-start_time).total_seconds()
        bench_results.append(BenchResult(
                operation='union', 
                tool_used='ogr', 
                params='',
                nb_cpu=multiprocessing.cpu_count(),
                secs_taken=secs_taken,
                secs_expected_one_cpu=4000,
                input1_filename=input1_path.name, 
                input2_filename=input2_path.name))
        
        print(f"Union with ogr ready in {secs_taken:.2f} secs")

    return bench_results

if __name__ == '__main__':
    
    # First make sure the testdata is present
    testdata_dir = get_testdata_dir()
    testdata_dir.mkdir(parents=True, exist_ok=True)
    prc2019_path = testdata_dir / 'Lbgbrprc19.gpkg'
    get_testdata_aiv(prc2019_path)
    prc2018_path = testdata_dir / 'Lbgbrprc18.gpkg'
    get_testdata_aiv(prc2018_path)

    # Now we can start benchmarking
    tmpdir = Path(tempfile.gettempdir()) / 'geofileops_benchmark'

    results = []
    results.extend(benchmark_buffer(tmpdir))
    results.extend(benchmark_intersect(tmpdir))
    results.extend(benchmark_union(tmpdir))
    
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
