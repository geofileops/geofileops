# -*- coding: utf-8 -*-
"""
Tests for functionalities in ogr_util.
"""

import datetime
import os
from pathlib import Path
import sys
import time 

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from geofileops.util import general_util

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

def test_processnice():
    # Test setting and getting some values for nice 
    # Remark: the nice values tests are spcifically written to accomodate for
    # windows specificalities: 
    #     - windows only supports 6 niceness classes. setprocessnice en 
    #       getprocessnice maps niceness values to these classes.
    #     - when setting REALTIME priority (-20 niceness) apparently this 
    #       results only to HIGH priority. 
    nice_orig = general_util.getprocessnice()
    for niceness in [-15, -10, 0, 10, 19]: 
        # Decreasing niceness (sometimes) isn't possible on linux, so skip 
        if os.name != 'nt':
            continue

        # Test!
        general_util.setprocessnice(niceness)
        nice = general_util.getprocessnice()
        assert nice == niceness

    # Test invalid values for nice value
    try:
        general_util.setprocessnice(20)    
        exception_raised = False
    except ValueError:
        exception_raised = True
    assert exception_raised is True
    try:
        general_util.setprocessnice(-21)    
        exception_raised = False
    except ValueError:
        exception_raised = True
    assert exception_raised is True
    
    # Reset niceness to original value before test
    general_util.setprocessnice(nice_orig)
