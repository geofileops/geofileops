import datetime
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))
import tempfile
from typing import List
import urllib.request
import zipfile

from geofileops.util import geofileops_gpd
from geofileops.util import geofileops_ogr
from geofileops import geofile

class BenchResult:
    def __init__(self, 
            operation: str,
            tool_used: str,
            secs_taken: float):
        self.operation = operation
        self.tool_used = tool_used
        self.secs_taken = secs_taken

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

    print('Convert shapefile to gpkg')
    geofile.convert(shp_paths[0], testdata_path)
    
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
    bench_results.append(
            BenchResult(operation='buffer', tool_used='geopandas', secs_taken=secs_taken))
    print(f"Buffer with geopandas ready in {secs_taken:.2f} secs")
    
    print('Start buffer with ogr')

    start_time = datetime.datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_ogr.gpkg"
    geofileops_ogr.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(
            BenchResult(operation='buffer', tool_used='ogr', secs_taken=secs_taken))

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
        bench_results.append(
                BenchResult(operation='intersect', tool_used='ogr', secs_taken=secs_taken))
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
        bench_results.append(
                BenchResult(operation='union', tool_used='ogr', secs_taken=secs_taken))
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
    #results.extend(benchmark_buffer(tmpdir))
    #results.extend(benchmark_intersect(tmpdir))
    results.extend(benchmark_union(tmpdir))

    for result in results:
        print(str(result))
