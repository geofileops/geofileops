# -*- coding: utf-8 -*-
"""
Tests for functionalities in ogr_util.
"""

from pathlib import Path
import sys

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

if __name__ == '__main__':
    # Init
    tmpdir = test_helper.init_test_for_debug(Path(__file__).stem)

    # Test...
    test_formatbytes()
