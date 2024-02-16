"""
Tests for functionalities in _general_util.
"""

import datetime
import os
import time

from geofileops.util import _general_util


def test_formatbytes():
    bytes_str = _general_util.formatbytes(1)
    assert bytes_str == "1.0 Byte"
    bytes_str = _general_util.formatbytes(2)
    assert bytes_str == "2.0 Bytes"
    bytes_str = _general_util.formatbytes(1024.0)
    assert bytes_str == "1.00 KB"
    bytes_str = _general_util.formatbytes(1024.0**2)
    assert bytes_str == "1.00 MB"
    bytes_str = _general_util.formatbytes(1024.0**3)
    assert bytes_str == "1.00 GB"
    bytes_str = _general_util.formatbytes(1024.0**4)
    assert bytes_str == "1.00 TB"


def test_format_progress():
    start_time = datetime.datetime.now()
    nb_todo = 10000
    for nb_done in range(0, nb_todo + 1, 2000):
        message = _general_util.format_progress(
            start_time=start_time,
            nb_done=nb_done,
            nb_todo=nb_todo,
            operation="test",
            nb_parallel=2,
        )
        time.sleep(0.5)
        if message is not None:
            print(message)


def test_TempEnv():
    test_var1 = "TEST_ENV_VARIABLE1"
    test_var2 = "TEST_ENV_VARIABLE2"
    assert test_var1 not in os.environ
    assert test_var2 not in os.environ
    os.environ[test_var2] = "test2_original_value"
    with _general_util.TempEnv(
        {test_var1: "test1_context_value", test_var2: "test2_context_value"}
    ):
        assert os.environ[test_var1] == "test1_context_value"
        assert os.environ[test_var2] == "test2_context_value"

    assert test_var1 not in os.environ
    assert os.environ[test_var2] == "test2_original_value"
