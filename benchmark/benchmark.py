# -*- coding: utf-8 -*-
"""
Module for benchmarking.
"""

from importlib import import_module
import logging
from pathlib import Path
import tempfile
    
import pandas as pd

import report

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def benchmark():
    # Init logging
    logging.basicConfig(
            format="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s", 
            datefmt="%H:%M:%S", level=logging.INFO)

    #Go!
    tmpdir = None
    #tmpdir = Path(r"C:\Temp") / 'geofileops_benchmark'
    
    # Discover and run all benchmark implementations
    tmp_dir = Path(tempfile.gettempdir()) / 'geobenchmark'
    logger.info(f"tmpdir: {tmp_dir}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    benchmarks_dir = Path(__file__).parent / "benchmarks"
    results = []
    for file in benchmarks_dir.glob("benchmark_*.py"):
        module_name = file.stem
        if (not module_name.startswith("_")) and (module_name not in globals()):
            benchmark_implementation = import_module(f"benchmarks.{module_name}", __package__)
            
            results.extend(benchmark_implementation.run(tmp_dir=tmp_dir))

    """
    # Write results to "latest" csv file
    results_latest_path = Path(__file__).resolve().parent / 'benchmark_results_latest.csv'
    results_dictlist = [vars(result) for result in results]
    results_df = pd.DataFrame(results_dictlist)
    results_df.to_csv(results_latest_path, index=False)
    """
    
    # Add results to general csv file
    results_path = Path(__file__).resolve().parent / 'benchmark_results.csv'
    results_dictlist = [vars(result) for result in results]
    results_df = pd.DataFrame(results_dictlist)
    if not results_path.exists():
        results_df.to_csv(results_path, index=False)
    else:
        results_df.to_csv(results_path, index=False, mode='a', header=False)

    # GEnerate reports
    report.report(results_path)

if __name__ == '__main__':
    benchmark()
