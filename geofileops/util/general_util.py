
import datetime
import logging
import math
import multiprocessing
import os
from typing import NamedTuple, Optional

import psutil

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def initgdal():
    # At least on windows, the init of this doensn't seem to work properly... 
    # TODO: should be solved somewhere else?
    if os.environ['PROJ_LIB'] is None:
        os.environ['PROJ_LIB'] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\proj"
        logger.warn(f"PROJ_LIB environment variable was not set, set to {os.environ['PROJ_LIB']}")
    if os.environ['GDAL_DATA'] is None:
        os.environ['GDAL_DATA'] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\gdal"
        logger.warn(f"GDAL_DATA environment variable was not set, set to {os.environ['GDAL_DATA']}")

def report_progress(
        start_time: datetime.datetime,
        nb_done: int,
        nb_todo: int,
        operation: Optional[str]):

    # 
    time_passed = (datetime.datetime.now()-start_time).total_seconds()
    if time_passed > 0 and nb_done > 0:
        processed_per_hour = (nb_done/time_passed) * 3600
        hours_to_go = (int)((nb_todo - nb_done)/processed_per_hour)
        min_to_go = (int)((((nb_todo - nb_done)/processed_per_hour)%1)*60)
        pct_progress = 100.0-(nb_todo-nb_done)*100/nb_todo
        print(f"\r{hours_to_go:3d}:{min_to_go:2d} left to do {operation} on {(nb_todo-nb_done):8d} of {nb_todo:8d} ({pct_progress:3.2f}%)    ", 
              end="", flush=True)

def formatbytes(bytes: float):
    """
    Return the given bytes as a human friendly KB, MB, GB, or TB string
    """

    bytes_float = float(bytes)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if bytes_float < KB:
        return '{0} {1}'.format(bytes_float,'Bytes' if 0 == bytes_float > 1 else 'Byte')
    elif KB <= bytes_float < MB:
        return '{0:.2f} KB'.format(bytes_float/KB)
    elif MB <= bytes_float < GB:
        return '{0:.2f} MB'.format(bytes_float/MB)
    elif GB <= bytes_float < TB:
        return '{0:.2f} GB'.format(bytes_float/GB)
    elif TB <= bytes_float:
        return '{0:.2f} TB'.format(bytes_float/TB)

ParallellisationParams = NamedTuple('result', [('nb_parallel', int), ('nb_batches_recommended', int), ('nb_rows_per_batch', int)])
def get_parallellisation_params(
        nb_rows_total: int,
        nb_parallel: int = -1,
        prev_nb_batches: int = None,
        verbose: bool = False) -> ParallellisationParams:
    """
    Determines recommended parallellisation params.

    Args:
        nb_rows_total (int): The total number of rows that will be processed
        nb_parallel (int, optional): The level of parallellisation requested. 
            If -1, tries to use all resources available. Defaults to -1.
        prev_nb_batches (int, optional): If applicable, the number of batches 
            used in a previous pass of the calculation. Defaults to None.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        ParallellisationParams (NamedTuple('result', [('nb_parallel', int), ('nb_batches_recommended', int), ('nb_rows_per_batch', int)])): The recommended parameters.
    """
    # Some initialisations
    bytes_basefootprint = 50*1024*1024   # Base footprint of a python process
    bytes_per_row = 100                  # Average memory needed per row in bytes. Remark: when running from VS code, 3 times higher!
    min_avg_rows_per_batch = 1000
    max_avg_rows_per_batch = 7500       # 60.000 on small test, but seems slow in larger
    bytes_min_per_process = bytes_basefootprint + bytes_per_row * min_avg_rows_per_batch
    bytes_usable = psutil.virtual_memory().available * 0.9
    
    # If the number of rows is really low, just use one batch
    # TODO: for very complex features, possibly this limit is not a good idea
    if nb_rows_total < min_avg_rows_per_batch:
        return ParallellisationParams(1, 1, nb_rows_total)

    if(nb_parallel == -1):
        nb_parallel = multiprocessing.cpu_count()

    # If the available memory is very small, check if we can use more swap 
    if bytes_usable < 1024*1024:
        bytes_usable = min(psutil.swap_memory().free, 1024*1024)
    logger.debug(f"memory_usable: {formatbytes(bytes_usable)} with mem.available: {formatbytes(psutil.virtual_memory().available)} and swap.free: {formatbytes(psutil.swap_memory().free)}") 

    # If not enough memory for the amount of parallellism asked, reduce
    if (nb_parallel * bytes_min_per_process) > bytes_usable:
        nb_parallel = int(bytes_usable/bytes_min_per_process)
        logger.debug(f"Nb_parallel reduced to {nb_parallel} to evade excessive memory usage")

    # Optimal number of batches and rows per batch based on memory usage
    nb_batches = math.ceil((nb_rows_total*bytes_per_row*nb_parallel)/(bytes_usable-bytes_basefootprint*nb_parallel))
    
    # Make sure the average batch doesn't contain > max_avg_rows_per_batch
    batch_size = math.ceil(nb_rows_total/nb_batches)
    if batch_size > max_avg_rows_per_batch:
        batch_size = max_avg_rows_per_batch
        nb_batches = math.ceil(nb_rows_total/batch_size)
    mem_predicted = (bytes_basefootprint + batch_size*bytes_per_row)*nb_batches

    # Make sure there are enough batches to use as much parallelism as possible
    if nb_batches > 1 and nb_batches < nb_parallel:
        if prev_nb_batches is None:
            nb_batches = round(nb_parallel*1.25)
        elif nb_batches < prev_nb_batches/4:
            nb_batches = round(nb_parallel*1.25)
    
    batch_size = math.ceil(nb_rows_total/nb_batches)

    # Log result
    if verbose:
        logger.info(f"nb_batches_recommended: {nb_batches}, rows_per_batch: {batch_size} for nb_rows_input_layer: {nb_rows_total} will result in mem_predicted: {formatbytes(mem_predicted)}")   

    return ParallellisationParams(nb_parallel, nb_batches, batch_size)
