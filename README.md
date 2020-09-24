# geofileops
Library to make spatial operations on geo files fast and easy. Accomplishes this by using geopandas, gdal and ogr2ogr under the hood to perform geospatial operations on files by using all available cores.

Remarks: 
* Early version, not feature complete yet, but quite some common operations should already be usable and are quite fast on large files (if sufficient cores available).
* Tested mainly on geopackage input files, but shapefiles should be ok as well. 

## Installation manual

1. Create and activate a new conda environment
```
conda create --name geofileops python=3.6 geopandas
conda activate geofileops
```
