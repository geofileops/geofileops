# -*- coding: utf-8 -*-
"""
Module containing some general utilities.
"""

import datetime
import logging
import os
from typing import Optional

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
        operation: Optional[str] = None,
        nb_parallel: int = 1):

    # If logging level not enabled for INFO, no progress reporting...
    if logger.isEnabledFor(logging.INFO) is False:
        return

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

def process_nice_to_priority_class(nice_value: int) -> int:
    if nice_value <= -15:
        return psutil.REALTIME_PRIORITY_CLASS
    elif nice_value <= -10:
        return psutil.HIGH_PRIORITY_CLASS
    elif nice_value <= -5:
        return psutil.ABOVE_NORMAL_PRIORITY_CLASS
    elif nice_value <= 0:
        return psutil.NORMAL_PRIORITY_CLASS
    elif nice_value <= 10:
        return psutil.BELOW_NORMAL_PRIORITY_CLASS
    else:
        return psutil.IDLE_PRIORITY_CLASS

def setprocessnice(nice_value: int):
    p = psutil.Process(os.getpid())
    if os.name == 'nt':
        p.nice(process_nice_to_priority_class(nice_value))
    else:
        p.nice(nice_value)

def getprocessnice() -> int:
    p = psutil.Process(os.getpid())
    nice_value = p.nice()
    if os.name == 'nt':
        if nice_value == psutil.REALTIME_PRIORITY_CLASS:
            return -20
        elif nice_value == psutil.HIGH_PRIORITY_CLASS:
            return -10
        elif nice_value == psutil.ABOVE_NORMAL_PRIORITY_CLASS:
            return -5
        elif nice_value == psutil.NORMAL_PRIORITY_CLASS:
            return 0
        elif nice_value == psutil.BELOW_NORMAL_PRIORITY_CLASS:
            return 10
        elif nice_value == psutil.IDLE_PRIORITY_CLASS:
            return 20
        else:
            return 0
    else:
        return int(nice_value)

def initialize_worker():
    # We don't want the workers to block the entire system, so nice them 
    setprocessnice(15)
