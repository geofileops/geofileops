# geofileops 

[![Actions Status](https://github.com/geofileops/geofileops/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/geofileops/geofileops/actions/workflows/tests.yml?query=workflow%3ATests) 
[![Coverage Status](https://codecov.io/gh/geofileops/geofileops/branch/main/graph/badge.svg)](https://codecov.io/gh/geofileops/geofileops)
[![PyPI version](https://img.shields.io/pypi/v/geofileops.svg)](https://pypi.org/project/geofileops)
[![Conda version](https://anaconda.org/conda-forge/geofileops/badges/version.svg)](https://anaconda.org/conda-forge/geofileops)

Library to make spatial operations on large geo files fast(er) and easy.

Remarks: 
* Most typical operations are available: 
  [buffer](https://geofileops.readthedocs.io/en/stable/api/geofileops.apply.html#geofileops.buffer),
  [simplify](https://geofileops.readthedocs.io/en/stable/api/geofileops.apply.html#geofileops.simolify),
  [dissolve](https://geofileops.readthedocs.io/en/stable/api/geofileops.apply.html#geofileops.dissolve),
  [union](https://geofileops.readthedocs.io/en/stable/api/geofileops.apply.html#geofileops.union),
  [erase](https://geofileops.readthedocs.io/en/stable/api/geofileops.apply.html#geofileops.erase)/difference, 
  [intersection](https://geofileops.readthedocs.io/en/stable/api/geofileops.apply.html#geofileops.intersection),...
* Any python function can be applied to a geofile in parallel using [apply](https://geofileops.readthedocs.io/en/stable/api/geofileops.apply.html#geofileops.apply)
* The speed (improvement) depends on the operation, the number of available cores and the size of the input files.
  * For CPU bound operations (eg. union,... between large input files) the processing time will depend on the number of available CPU cores. For (very) large files the typical processing time can be divided by the number of available cores.
  * For dissolve on (very) large files, the speed improvement can be more than the processing time divided by the available cores.
* Tested on geopackage and shapefile input/output files. However, geopackage is highly recommended as it will offer better performance in geofileops... and also for the reasons listed here: www.switchfromshapefile.org.

Documentation on how to use geofileops can be found [here](https://geofileops.readthedocs.io).

The following chart gives an impression of the speed improvement that can be expected when processing larger files (including I/O!). More information about this benchmark can be found [here](https://github.com/geofileops/geobenchmark).

![Geo benchmark](https://github.com/geofileops/geobenchmark/blob/main/results_vector_ops/GeoBenchmark.png)
