from benchmark import benchmarker


def main():
    # Run the benchmark function(s)
    functions_to_run = [
        # "clip",
        # "intersection",
        "intersection_complexpoly_agri",
        # "intersection_gridsize",
        # "symmetric_difference_complexpoly_agri",
    ]
    # Run all bechmark functions
    # functions_to_run = None
    benchmarker.run_benchmarks(["benchmarks_geofileops"], functions_to_run)


if __name__ == "__main__":
    main()
