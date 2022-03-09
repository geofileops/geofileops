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
    # The nice values tests are spcifically written to accomodate for  
    # windows specificalities: 
    #     - windows only supports 6 niceness classes. setprocessnice en 
    #       getprocessnice maps niceness values to these classes.
    #     - when setting REALTIME priority (-20 niceness) apparently this 
    #       results only to HIGH priority. 
    for niceness in [-15, -10, 0, 10, 20]:    
        general_util.setprocessnice(niceness)
        nice = general_util.getprocessnice()
        assert nice == niceness

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Test...
    #test_formatbytes()
    #test_format_progress()
    test_processnice()