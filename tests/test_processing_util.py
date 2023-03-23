# -*- coding: utf-8 -*-
"""
Tests for functionalities in _processing_util.
"""

import os

import psutil
import pytest

from geofileops.util import _processing_util


def test_processnice():
    # Test setting and getting some values for nice
    # Remark: the nice values tests are spcifically written to accomodate for
    # windows specificalities:
    #     - windows only supports 6 niceness classes. setprocessnice en
    #       getprocessnice maps niceness values to these classes.
    #     - when setting REALTIME priority (-20 niceness) apparently this
    #       results only to HIGH priority.
    nice_orig = _processing_util.getprocessnice()
    for niceness in [-15, -10, 0, 10, 19]:
        # Decreasing niceness (sometimes) isn't possible on linux, so skip
        if os.name != "nt":
            continue

        # Test!
        _processing_util.setprocessnice(niceness)
        nice = _processing_util.getprocessnice()
        assert nice == niceness

    # Test invalid values for nice value
    try:
        _processing_util.setprocessnice(20)
        exception_raised = False
    except ValueError:
        exception_raised = True
    assert exception_raised is True
    try:
        _processing_util.setprocessnice(-21)
        exception_raised = False
    except ValueError:
        exception_raised = True
    assert exception_raised is True

    # Reset niceness to original value before test
    _processing_util.setprocessnice(nice_orig)


@pytest.mark.skipif(os.name != "nt", reason="run only on windows")
@pytest.mark.parametrize(
    "nice_value, expected_priorityclass",
    [
        (-20, psutil.REALTIME_PRIORITY_CLASS),
        (-15, psutil.HIGH_PRIORITY_CLASS),
        (-10, psutil.ABOVE_NORMAL_PRIORITY_CLASS),
        (0, psutil.NORMAL_PRIORITY_CLASS),
        (10, psutil.BELOW_NORMAL_PRIORITY_CLASS),
        (15, psutil.IDLE_PRIORITY_CLASS),
    ],
)
def test_process_nice_to_priorityclass(nice_value, expected_priorityclass):
    assert (
        _processing_util.process_nice_to_priorityclass(nice_value)
        == expected_priorityclass
    )


@pytest.mark.skipif(os.name != "nt", reason="run only on windows")
@pytest.mark.parametrize(
    "priorityclass, expected_nice",
    [
        (psutil.REALTIME_PRIORITY_CLASS, -20),
        (psutil.HIGH_PRIORITY_CLASS, -15),
        (psutil.ABOVE_NORMAL_PRIORITY_CLASS, -10),
        (psutil.NORMAL_PRIORITY_CLASS, 0),
        (psutil.BELOW_NORMAL_PRIORITY_CLASS, 10),
        (psutil.IDLE_PRIORITY_CLASS, 19),
    ],
)
def test_process_priorityclass_to_nice(priorityclass, expected_nice):
    assert (
        _processing_util.process_priorityclass_to_nice(priorityclass) == expected_nice
    )
