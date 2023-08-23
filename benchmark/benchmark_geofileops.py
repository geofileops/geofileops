from benchmark import benchmarker


def main():
    # benchmarker.run_benchmarks(["benchmarks_geofileops"])

    # return

    # Only run specific benchmark function(s)
    benchmarker.run_benchmarks(["benchmarks_geofileops"], ["intersection_complexpoly"])


if __name__ == "__main__":
    main()
