# -*- coding: utf-8 -*-
"""
Module to benchmark geopandas operations.
"""

from datetime import datetime
import logging
from pathlib import Path

import geopandas as gpd

from benchmarker import RunResult
from benchmarks import testdata

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def buffer(tmp_dir: Path) -> RunResult:
    
    ### Init ###
    input_path, _ = testdata.get_testdata(tmp_dir)
    
    ### Go! ###
    # Read input file
    start_time = datetime.now()
    gdf = gpd.read_file(input_path)
    logger.info(f"time for read: {(datetime.now()-start_time).total_seconds()}")
    
    # Buffer
    start_time_buffer = datetime.now()
    gdf.geometry = gdf.geometry.buffer(distance=1, resolution=5)
    logger.info(f"time for buffer: {(datetime.now()-start_time_buffer).total_seconds()}")
    
    # Write to output file
    start_time_write = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_geopandas_buf.gpkg"
    # This read actually used pyogrio, so is not really geopandas
    gdf.to_file(output_path, layer=output_path.stem, driver="GPKG")
    logger.info(f"write took {(datetime.now()-start_time_write).total_seconds()}")    
    result = RunResult(
            package="geopandas", 
            package_version=gpd.__version__,
            operation="buffer", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="buffer agri parcels BEFL (~500.000 polygons)")
    
    # Cleanup
    output_path.unlink()

    return result

def dissolve(tmp_dir: Path) -> RunResult:
    
    ### Init ###
    input_path, _ = testdata.get_testdata(tmp_dir)
    
    ### Go! ###
    # Read input file
    start_time = datetime.now()
    gdf = gpd.read_file(input_path)
    logger.info(f"time for read: {(datetime.now()-start_time).total_seconds()}")
    
    # dissolve
    start_time_dissolve = datetime.now()
    gdf = gdf.dissolve()
    logger.info(f"time for dissolve: {(datetime.now()-start_time_dissolve).total_seconds()}")
    
    # Write to output file
    start_time_write = datetime.now()
    output_path = tmp_dir / f"{input_path.stem}_geopandas_diss.gpkg"
    gdf.to_file(output_path, layer=output_path.stem, driver="GPKG")
    logger.info(f"write took {(datetime.now()-start_time_write).total_seconds()}")
    result = RunResult(
            package="geopandas", 
            package_version=gpd.__version__,
            operation="dissolve", 
            secs_taken=(datetime.now()-start_time).total_seconds(),
            operation_descr="dissolve agri parcels BEFL (~500.000 polygons)")
    
    # Cleanup and return
    output_path.unlink()
    return result

def intersect(tmp_dir: Path) -> RunResult:
    
    ### Init ###
    input1_path, input2_path = testdata.get_testdata(tmp_dir)
        
    ### Go! ###
    # Read input files
    start_time = datetime.now()
    input1_gdf = gpd.read_file(input1_path)
    input2_gdf = gpd.read_file(input2_path)
    logger.info(f"time for read: {(datetime.now()-start_time).total_seconds()}")
    
    # intersect
    start_time_intersect = datetime.now()
    output_gdf = input1_gdf.overlay(input2_gdf, how="intersection")
    logger.info(f"time for intersect: {(datetime.now()-start_time_intersect).total_seconds()}")

    # Write to output file
    start_time_write = datetime.now()
    output_path = tmp_dir / f"{input1_path.stem}_inters_{input2_path.stem}.gpkg"
    output_gdf.to_file(output_path, layer=output_path.stem, driver="GPKG")
    logger.info(f"write took {(datetime.now()-start_time_write).total_seconds()}")
    secs_taken = (datetime.now()-start_time).total_seconds()
    result = RunResult(
            package="geopandas", 
            package_version=gpd.__version__,
            operation='intersect', 
            secs_taken=secs_taken,
            operation_descr="intersect between 2 agri parcel layers BEFL (2*~500.000 polygons)")
    
    # Cleanup and return
    output_path.unlink()
    return result
