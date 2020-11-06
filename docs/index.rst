.. geofileops documentation master file, created by
   sphinx-quickstart on Thu Nov  5 20:17:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

geofileops |version|
====================

This python library makes spatial operations on geo files fast and easy. 

It provides an easy to use API, and builds on geopandas, gdal and ogr2ogr. 
For large files, all geospatial operations become fast(er) because 
all available cores can be used.

Most typical operations are available: buffer, simplify, dissolve, union, 
erase, intersect,...
Tested on geopackage and shapefile input files.

.. toctree::
   :maxdepth: 2

   Home <self>
   Getting started <getting_started>
   User guide <user_guide>
   API reference <reference>
   Development <development>
   Release notes <release_notes>
