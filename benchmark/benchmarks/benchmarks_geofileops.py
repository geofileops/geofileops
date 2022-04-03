# -*- coding: utf-8 -*-
"""
Module to benchmark geofileops operations.
"""

from datetime import datetime
import logging
import multiprocessing
from pathlib import Path
import sys

from benchmarker import RunResult
from benchmarks import testdata

# Add path so the benchmark packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import geofileops as gfo
from geofileops.util import _geoops_sql
from geofileops.util import _geoops_gpd

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

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
    gfo.buffer(input_path, output_path, distance=1, force=True)
    result = RunResult(
            package=_get_package(), 
            package_version=_get_version(),
            operation="buffer", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="buffer on agri parcel layer BEFL (~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()})
    
    # Cleanup and return
    output_path.unlink()
    return result

def buffer_spatialite(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    
    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_spatialite.gpkg"
    _geoops_sql.buffer(input_path, output_path, distance=1, force=True)
    result = RunResult(
            package=_get_package(), 
            package_version=_get_version(),
            operation="buffer_spatialite", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="buffer on agri parcel layer BEFL (~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()})
    
    # Cleanup and return
    output_path.unlink()
    return result

def buffer_gpd(tmp_dir: Path) -> RunResult:
    # Init
    input_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    
    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_buf_gpd.gpkg"
    _geoops_gpd.buffer(input_path, output_path, distance=1, force=True)
    result = RunResult(
            package=_get_package(), 
            package_version=_get_version(),
            operation="buffer_gpd", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="buffer on agri parcel layer BEFL (~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()})
    
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
            force=True)
    result = RunResult(
            package=_get_package(), 
            package_version=_get_version(),
            operation="dissolve",
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="dissolve on agri parcels BEFL (~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()})

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
            force=True)
    result = RunResult(
            package=_get_package(), 
            package_version=_get_version(),
            operation="dissolve_groupby", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="dissolve on agri parcels BEFL (~500.000 polygons), groupby=[GEWASGROEP]",
            run_details={"nb_cpu": multiprocessing.cpu_count()})
    
    # Cleanup and return
    output_path.unlink()
    return result

def clip(tmp_dir: Path) -> RunResult:
    # Init
    input1_path, input2_path = testdata.get_testdata(tmp_dir)
    output_path = tmp_dir / f"{input1_path.stem}_clip_{input2_path.stem}.gpkg"
    
    # Go!
    start_time = datetime.now()
    gfo.clip(
            input_path=input1_path, 
            clip_path=input2_path, 
            output_path=output_path,
            force=True)
    result = RunResult(
            package="geofileops", 
            package_version=gfo.__version__,
            operation='clip', 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="clip between 2 agri parcel layers BEFL (2*~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()})

    # Cleanup and return
    #output_path.unlink()
    return result

def intersect(tmp_dir: Path) -> RunResult:
    # Init
    input1_path = testdata.TestFile.AGRIPRC_2018.get_file(tmp_dir)
    input2_path = testdata.TestFile.AGRIPRC_2019.get_file(tmp_dir)
    
    # Go!
    start_time = datetime.now()
    output_path = tmp_dir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    gfo.intersect(
            input1_path=input1_path, 
            input2_path=input2_path, 
            output_path=output_path,
            force=True)
    result = RunResult(
            package=_get_package(), 
            package_version=_get_version(),
            operation='intersect', 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="intersect between 2 agri parcel layers BEFL (2*~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()})

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
            force=True)
    result = RunResult(
            package=_get_package(), 
            package_version=_get_version(),
            operation="union",
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="union between 2 agri parcel layers BEFL (2*~500.000 polygons)",
            run_details={"nb_cpu": multiprocessing.cpu_count()})

    # Cleanup and return
    output_path.unlink()
    return result
    