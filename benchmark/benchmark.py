import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))
import tempfile
from typing import List

from geofileops.util import geofileops_gpd
from geofileops.util import geofileops_ogr

class BenchResult:
    def __init__(self, 
            operation: str,
            tool_used: str,
            secs_taken: float):
        self.operation = operation
        self.tool_used = tool_used
        self.secs_taken = secs_taken

def get_testdata_dir() -> Path:
    #return Path(__file__).resolve().parent / 'data'
    testdata_dir = Path(tempfile.gettempdir()) / 'geofileops_benchmark_data'
    return testdata_dir

def benchmark_buffer(tmp_dir: Path) -> List[BenchResult]:
    bench_results = []
    input_path = get_testdata_dir() / 'Lbgbrprc18.gpkg'
    output_path = tmp_dir / f"{input_path.stem}_buf{input_path.suffix}"
    
    print('Start buffer with geopandas')
    start_time = datetime.datetime.now()
    geofileops_gpd.buffer(input_path, output_path, distance=1, force=True)
    secs_taken = (datetime.datetime.now()-start_time).total_seconds()
    bench_results.append(
            BenchResult(operation='buffer', tool_used='geopandas', secs_taken=secs_taken))
    print(f"Buffer with geopandas ready in {secs_taken:.2f} secs")
    '''
    print('Start buffer with ogr')
    import os
    os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
    try:
        start_time = datetime.datetime.now()
        geofileops_ogr.buffer(input_path, output_path, distance=1, force=True)
        secs_taken = (datetime.datetime.now()-start_time).total_seconds()
        bench_results.append(
                BenchResult(operation='buffer', tool_used='ogr2ogr', secs_taken=secs_taken))
    finally:
        del os.environ['GDAL_BIN']
    print(f"Buffer with ogr ready in {secs_taken:.2f} secs")
    '''

    return bench_results

if __name__ == '__main__':
    
    tmpdir = Path(tempfile.gettempdir()) / 'geofileops_benchmark'
    result = benchmark_buffer(tmpdir)
    print(str(result))
