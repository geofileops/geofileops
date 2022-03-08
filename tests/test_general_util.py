# -*- coding: utf-8 -*-
"""
Tests for functionalities in ogr_util.
"""

import datetime
from pathlib import Path
import psutil
import sys
import time 

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import general_util
import test_helper

def test_formatbytes():
    bytes_str = general_util.formatbytes(1)
    assert bytes_str == '1.0 Byte'
    bytes_str = general_util.formatbytes(2)
    assert bytes_str == '2.0 Bytes'
    bytes_str = general_util.formatbytes(1024.0)
    assert bytes_str == '1.00 KB'
    bytes_str = general_util.formatbytes(1024.0 ** 2)
    assert bytes_str == '1.00 MB'
    bytes_str = general_util.formatbytes(1024.0 ** 3)
    assert bytes_str == '1.00 GB'
    bytes_str = general_util.formatbytes(1024.0 ** 4)
    assert bytes_str == '1.00 TB'
    
def test_getprocessnice():
    nice = general_util.getprocessnice()
    assert nice >= -20 and nice <= 20

def test_process_nice_to_priority_class():
    priority_class = general_util.process_nice_to_priority_class(-15)
    assert priority_class == psutil.REALTIME_PRIORITY_CLASS
    priority_class = general_util.process_nice_to_priority_class(-10)
    assert priority_class == psutil.HIGH_PRIORITY_CLASS
    priority_class = general_util.process_nice_to_priority_class(-5)
    assert priority_class == psutil.ABOVE_NORMAL_PRIORITY_CLASS
    priority_class = general_util.process_nice_to_priority_class(0)
    assert priority_class == psutil.NORMAL_PRIORITY_CLASS
    priority_class = general_util.process_nice_to_priority_class(10)
    assert priority_class == psutil.BELOW_NORMAL_PRIORITY_CLASS
    priority_class = general_util.process_nice_to_priority_class(11)
    assert priority_class == psutil.IDLE_PRIORITY_CLASS

def test_format_progress():
    start_time = datetime.datetime.now()
    nb_todo = 10000
    for nb_done in range(0, nb_todo+1, 2000):
        message = general_util.format_progress(
                start_time=start_time, 
                nb_done=nb_done, 
                nb_todo=nb_todo, 
                operation="test", 
                nb_parallel=2)
        time.sleep(0.5)
        if message is not None:
            print(message)

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Test...
    #test_formatbytes()
    test_format_progress()