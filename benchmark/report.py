# -*- coding: utf-8 -*-
"""
Module for benchmarking.
"""

from pathlib import Path

import numpy as np
import pandas as pd 

from util import charting_util

def report(results_path: Path):
    
    benchmark_df = pd.read_csv(results_path)
    reports_dir = Path(__file__).resolve().parent / "reports"

    # Convert input file to new format
    '''
    results_path = Path(__file__).resolve().parent / "benchmark_results_old.csv"
    benchmark_df = pd.read_csv(results_path)
    reports_dir = Path(__file__).resolve().parent / "reports"
    results_out_path = Path(__file__).resolve().parent / "benchmark_results.csv"
    benchmark_df = benchmark_df.rename(columns={"package_version": "version"})
    benchmark_df["package"] = "geofileops"
    #benchmark_df["tool_used"] = "lib=" + benchmark_df["tool_used"].replace("sql", "spatialite").replace("ogr", "spatialite")
    benchmark_df["nb_cpu"] = "'nb_cpu_used': " + benchmark_df["nb_cpu"].astype(str).replace("nan", np.nan)
    benchmark_df["input1_filename"] = ", 'input1': '" + benchmark_df["input1_filename"] + "'"
    benchmark_df["input2_filename"] = ", 'input2': '" + benchmark_df["input2_filename"] + "'"
    benchmark_df["params"] = ", 'params': '" + benchmark_df["params"] + "'"
    benchmark_df["run_details"] = ("{" + 
            benchmark_df["nb_cpu"].astype(str).replace("nan", "") +
            benchmark_df["input1_filename"].astype(str).replace("nan", "") + 
            benchmark_df["input2_filename"].astype(str).replace("nan", "") + 
            benchmark_df["params"].astype(str).replace("nan", "") + "}")
    benchmark_df = benchmark_df[["datetime", "package", "version", "operation", "secs_taken", "run_details"]]
    benchmark_df.to_csv(results_out_path, index=False)
    '''

    ### Detailed report per package and per operation ###
    for package in benchmark_df["package"].unique():
        reports_package_dir = reports_dir / package
        reports_package_dir.mkdir(parents=True, exist_ok=True)

        package_df = benchmark_df.loc[benchmark_df["package"] == package]
        for operation in package_df["operation"].unique():
            package_operation_df = package_df.loc[benchmark_df["operation"] == operation]
            package_operation_df = package_operation_df[["version", "secs_taken"]]
            package_operation_df = package_operation_df.set_index(["version"])
            results_report_path = reports_package_dir / f"{package}_{operation}.png"
            charting_util.save_chart(
                    df=package_operation_df,
                    title=f"{package}-{operation}", 
                    output_path=results_report_path)

    benchmark_maxversions_df = (benchmark_df[["package", "operation", "version"]]
            .sort_values(["package", "operation", "version"], ascending=False)
            .groupby(["package", "operation"])
            .first()
            .reset_index()
            .set_index(["package", "operation", "version"]))
    benchmark_maxversion_df = benchmark_df.set_index(["package", "operation", "version"])
    benchmark_maxversion_df = (benchmark_maxversion_df
            .loc[benchmark_maxversion_df.index.isin(benchmark_maxversions_df.index)]
            .reset_index())[["package", "operation", "secs_taken"]]
    benchmark_maxversion_df = benchmark_maxversion_df.pivot_table(index="operation", columns="package")
    results_report_path = reports_dir / f"GeoBenchmark.png"
    charting_util.save_chart(
            df=benchmark_maxversion_df,
            title="Geo benchmark", 
            output_path=results_report_path,
            linestyle="None")

if __name__ == "__main__":

    results_path = Path(__file__).resolve().parent / 'benchmark_results.csv'
    report(results_path)
