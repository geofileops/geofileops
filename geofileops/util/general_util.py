
import datetime
import logging
import multiprocessing
import os

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
        operation: str):

    # 
    time_passed = (datetime.datetime.now()-start_time).total_seconds()
    if time_passed > 0 and nb_done > 0:
        processed_per_hour = (nb_done/time_passed) * 3600
        hours_to_go = (int)((nb_todo - nb_done)/processed_per_hour)
        min_to_go = (int)((((nb_todo - nb_done)/processed_per_hour)%1)*60)
        print(f"\r{hours_to_go:3d}:{min_to_go:2d} left to do {operation} on {(nb_todo-nb_done):6d} of {nb_todo}", 
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

def get_parallellisation_params(
        nb_rows_total: int,
        nb_parallel: int = -1,
        verbose: bool = False):
    # Some initialisations
    memory_basefootprint = 50*1024*1024
    memory_per_row = 100*1024*1024/30000   # Memory usage per row
    min_rows_per_batch = 5000
    memory_min_per_process = memory_basefootprint + memory_per_row * min_rows_per_batch
    memory_usable = psutil.virtual_memory().available * 0.9
    

    if(nb_parallel == -1):
        if nb_rows_total < min_rows_per_batch:
            nb_parallel = 1
        else:
            nb_parallel = multiprocessing.cpu_count()

    # If the available memory is very small, check if we can use more swap 
    if memory_usable < 1024*1024:
        memory_usable = min(psutil.swap_memory().free, 1024*1024)
    logger.info(f"memory_usable: {formatbytes(memory_usable)} with mem.available: {formatbytes(psutil.virtual_memory().available)} and swap.free: {formatbytes(psutil.swap_memory().free)}") 

    # If not enough memory for the amount of parallellism asked, reduce
    if (nb_parallel * memory_min_per_process) > memory_usable:
        nb_parallel = int(memory_usable/memory_min_per_process)
        logger.info(f"Nb_parallel reduced to {nb_parallel} to evade excessive memory usage")

    # Optimal number of batches and rows per batch 
    nb_batches = int((nb_rows_total*memory_per_row*nb_parallel)/(memory_usable-memory_basefootprint*nb_parallel))
    if nb_batches < nb_parallel:
        nb_batches = nb_parallel
    
    batch_size = int(nb_rows_total/nb_batches)
    mem_predicted = (memory_basefootprint + batch_size*memory_per_row)*nb_batches

    if verbose:
        logger.info(f"nb_batches: {nb_batches}, rows_per_batch: {batch_size} for nb_rows_input_layer: {nb_rows_total} will result in mem_predicted: {formatbytes(mem_predicted)}")   

    return (nb_parallel, nb_batches, batch_size)
      