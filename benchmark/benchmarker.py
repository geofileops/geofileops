# -*- coding: utf-8 -*-
"""
Module for benchmarking.
"""

import datetime
from importlib import import_module
import logging
from pathlib import Path
import sys
import tempfile
from typing import List, Optional

import pandas as pd

# Add path so the local geofileops packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  
import report

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

class RunResult:
    """ The result of a benchmark run. """
    def __init__(self, 
            package: str,
            package_version: str,
            operation: str,
            operation_descr: str,
            secs_taken: float,
            run_details: Optional[dict] = None):
        """
        Constructor for a RunResult.

        Args:
            package (str): Package being benchmarked.
            package_version (str): Version of the package.
            operation (str): Operation name.
            operation_descr (str): Description of the operation.
            secs_taken (float): Seconds the operation took.
            run_details (dict, optional): (Important) details of this specific
                run with impact on performance. Eg. # CPU's used,...
        """
        self.run_datetime = datetime.datetime.now()
        self.package = package
        self.package_version = package_version
        self.operation = operation
        self.operation_descr = operation_descr
        self.secs_taken = secs_taken
        self.run_details = run_details
        
    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"

def run_benchmarks(
        benchmarks: Optional[List[str]] = None):
    # Init logging
    logging.basicConfig(
            format="%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s", 
            datefmt="%H:%M:%S", level=logging.INFO)

    # Discover and run all benchmark implementations
    tmp_dir = Path(tempfile.gettempdir()) / "geobenchmark"
    logger.info(f"tmpdir: {tmp_dir}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    benchmarks_dir = Path(__file__).parent / "benchmarks"
    results = []
    for file in benchmarks_dir.glob("benchmark_*.py"):
        module_name = file.stem
        if (not module_name.startswith("_")) and (module_name not in globals()):
            if benchmarks is None or module_name in benchmarks:
                benchmark_implementation = import_module(f"benchmarks.{module_name}", __package__)
                results.extend(benchmark_implementation.run(tmp_dir=tmp_dir))
    
    # Add results to csv file
    results_path = Path(__file__).resolve().parent / "benchmark_results.csv"
    results_dictlist = [vars(result) for result in results]
    results_df = pd.DataFrame(results_dictlist)
    if not results_path.exists():
        results_df.to_csv(results_path, index=False)
    else:
        results_df.to_csv(results_path, index=False, mode="a", header=False)

    # Generate reports
    output_dir = Path(__file__).resolve().parent / "reports"
    report.generate_reports(results_path, output_dir)

if __name__ == "__main__":
    run_benchmarks()
