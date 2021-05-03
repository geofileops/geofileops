# geofileops
Library to make spatial operations on large geo files fast(er) and easy. Accomplishes this by using geopandas, gdal and sqlite3/spatialite under the hood to perform geospatial operations on files by using all available cores.

Remarks: 
* Most typical operations are available: buffer, simplify, dissolve, union, erase, intersect,...
* Only/mainly faster on large files and for CPU intensive operations (eg. overlays). The more available cores, the faster. 
* Tested on geopackage and shapefile input files.

Documentation can be found here:
https://geofileops.readthedocs.io/en/latest/index.html
