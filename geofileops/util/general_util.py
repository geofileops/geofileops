# -*- coding: utf-8 -*-
"""
Module containing some general utilities.
"""

import datetime
import logging
import math
import multiprocessing
from typing import NamedTuple

import psutil

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

class MissingRuntimeDependencyError(Exception):
    """
    Exception raised when an unsupported SQL statement is passed.

    Attributes:
        message (str): Exception message
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

################################################################################
# The real work
################################################################################

def report_progress(
        start_time: datetime.datetime,
        nb_done: int,
        nb_todo: int,
        operation: str = None,
        nb_parallel: int = 1):

    # Init
    time_passed = (datetime.datetime.now()-start_time).total_seconds()
    pct_progress = 100.0-(nb_todo-nb_done)*100/nb_todo
    
    # If we haven't really started yet, don't report time estimate yet
    if nb_done == 0:
        print(f"\r  ?: ? left to do {operation} on {(nb_todo-nb_done):8d} of {nb_todo:8d} ({pct_progress:3.2f}%)    ", 
              end="", flush=True)
    elif time_passed > 0:
        # Else, report progress properly...
        processed_per_hour = (nb_done/time_passed) * 3600
        # Correct the nb processed per hour if running parallel 
        if nb_done < nb_parallel:
            processed_per_hour = round(processed_per_hour * nb_parallel / nb_done)
        hours_to_go = (int)((nb_todo - nb_done)/processed_per_hour)
        min_to_go = (int)((((nb_todo - nb_done)/processed_per_hour)%1)*60)
        pct_progress = 100.0-(nb_todo-nb_done)*100/nb_todo
        if pct_progress < 100:
            print(f"\r{hours_to_go:3d}:{min_to_go:2d} left to do {operation} on {(nb_todo-nb_done):8d} of {nb_todo:8d} ({pct_progress:3.2f}%)    ", 
                  end="", flush=True)
        else:
            print(f"\r{hours_to_go:3d}:{min_to_go:2d} left to do {operation} on {(nb_todo-nb_done):8d} of {nb_todo:8d} ({pct_progress:3.2f}%)    \n", 
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
        return '{0} {1}'.format(bytes_float, 'Bytes' if bytes_float > 1 else 'Byte')
    elif KB <= bytes_float < MB:
        return '{0:.2f} KB'.format(bytes_float/KB)
    elif MB <= bytes_float < GB:
        return '{0:.2f} MB'.format(bytes_float/MB)
    elif GB <= bytes_float < TB:
        return '{0:.2f} GB'.format(bytes_float/GB)
    elif TB <= bytes_float:
        return '{0:.2f} TB'.format(bytes_float/TB)

class ParallelizationConfig():
    def __init__(self, 
            bytes_basefootprint: int = 50*1024*1024, 
            bytes_per_row: int = 100, 
            min_avg_rows_per_batch: int = 1000, 
            max_avg_rows_per_batch: int = 10000, 
            bytes_min_per_process = None, 
            bytes_usable = None):
        self.bytes_basefootprint = bytes_basefootprint
        self.bytes_per_row = bytes_per_row
        self.min_avg_rows_per_batch = min_avg_rows_per_batch
        self.max_avg_rows_per_batch = max_avg_rows_per_batch
        if bytes_min_per_process is None:
            self.bytes_min_per_process = bytes_basefootprint + bytes_per_row * min_avg_rows_per_batch
        else:
            self.bytes_min_per_process = bytes_min_per_process
        if bytes_usable is None: 
            self.bytes_usable = psutil.virtual_memory().available * 0.9
        else:
            self.bytes_usable = bytes_usable

parallelizationParams = NamedTuple('result', [('nb_parallel', int), ('nb_batches_recommended', int), ('nb_rows_per_batch', int)])
def get_parallelization_params(
        nb_rows_total: int,
        nb_parallel: int = -1,
        prev_nb_batches: int = None,
        parallelization_config: ParallelizationConfig = None,
        verbose: bool = False) -> parallelizationParams:
    """
    Determines recommended parallelization params.

    Args:
        nb_rows_total (int): The total number of rows that will be processed
        nb_parallel (int, optional): The level of parallelization requested. 
            If -1, tries to use all resources available. Defaults to -1.
        prev_nb_batches (int, optional): If applicable, the number of batches 
            used in a previous pass of the calculation. Defaults to None.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        parallelizationParams (NamedTuple('result', [('nb_parallel', int), ('nb_batches_recommended', int), ('nb_rows_per_batch', int)])): The recommended parameters.
    """
    # Init parallelization config

    # If config is None, set to empty dict
    if parallelization_config is not None:
        parallelization_config_local = parallelization_config
    else:
        parallelization_config_local = ParallelizationConfig()
    
    # If the number of rows is really low, just use one batch
    # TODO: for very complex features, possibly this limit is not a good idea
    if nb_rows_total < parallelization_config_local.min_avg_rows_per_batch:
        return parallelizationParams(1, 1, nb_rows_total)

    if(nb_parallel == -1):
        nb_parallel = multiprocessing.cpu_count()

    # If the available memory is very small, check if we can use more swap 
    if parallelization_config_local.bytes_usable < 1024*1024:
        bytes_usable = min(psutil.swap_memory().free, 1024*1024)
    logger.debug(f"memory_usable: {formatbytes(parallelization_config_local.bytes_usable)} with mem.available: {formatbytes(psutil.virtual_memory().available)} and swap.free: {formatbytes(psutil.swap_memory().free)}") 

    # If not enough memory for the amount of parallellism asked, reduce
    if (nb_parallel * parallelization_config_local.bytes_min_per_process) > parallelization_config_local.bytes_usable:
        nb_parallel = int(parallelization_config_local.bytes_usable/parallelization_config_local.bytes_min_per_process)
        logger.debug(f"Nb_parallel reduced to {nb_parallel} to evade excessive memory usage")

    # Optimal number of batches and rows per batch based on memory usage
    nb_batches = math.ceil(
            (nb_rows_total*parallelization_config_local.bytes_per_row*nb_parallel)/
            (parallelization_config_local.bytes_usable-parallelization_config_local.bytes_basefootprint*nb_parallel))
    
    # Make sure the average batch doesn't contain > max_avg_rows_per_batch
    batch_size = math.ceil(nb_rows_total/nb_batches)
    if batch_size > parallelization_config_local.max_avg_rows_per_batch:
        batch_size = parallelization_config_local.max_avg_rows_per_batch
        nb_batches = math.ceil(nb_rows_total/batch_size)
    mem_predicted = (parallelization_config_local.bytes_basefootprint + batch_size*parallelization_config_local.bytes_per_row)*nb_batches

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

    return parallelizationParams(nb_parallel, nb_batches, batch_size)
