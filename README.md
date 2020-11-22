# geofileops
Library to make spatial operations on geo files fast and easy. Accomplishes this by using geopandas, gdal and ogr2ogr under the hood to perform geospatial operations on files by using all available cores.

Remarks: 
* Most typical operations are available: buffer, simplify, dissolve, union, erase, intersect,...
* Quite fast on large files. The more available cores, the faster obviously.
* Tested on geopackage and shapefile input files.

Documentation can be found here:
https://geofileops.readthedocs.io/en/latest/index.html
