# geofileops
Library to make spatial operations on large geo files fast(er) and easy.

Remarks: 
* Most typical operations are available: buffer, simplify, dissolve, union, erase, intersect,...
* The speed (improvement) depends on the number of available cores, the size of the input files and whether the operation is CPU intensive.
  * For CPU bound operations (eg. intersects,... between large input files) the processing time can be divided by the number of available cores.
  * For dissolve on (very) large files, the speed improvement might be a lot faster, even more than the processing time divided by the available cores.
  * For small files and/or computationally easy operations (eg. buffer) geofileops might be slower than other libraries.
* Tested on geopackage and shapefile input files.

Documentation can be found here:
https://geofileops.readthedocs.io/en/latest/index.html
