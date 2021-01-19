User guide
==========
The main objective of GeofileOps is to provide a high-level API to do spatial 
operations on GIS files like GeoPackages or shapefiles in a fast way. 
Most libraries only use one CPU to to spatial operations. GeofileOps use all
available CPU's, so especially for large file processing time should be the 
typical expected time divided by the number of available CPU's. 

As a side product some functions that are used internally are also exposed in 
the general module. 

geofileops.geofileops
---------------------
The first type of operations are operations that are executed on one 
file/layer. Examples are buffer, simplify, dissolve,...

This is an example for a buffer operation:

.. code-block:: python

    from geofileops import geofileops
    
    geofileops.buffer(
            input_path='...',
            output_path='...',
            distance=2)

By default all available CPU's will be used. You can specify the 
number of CPU's to be used with the nb_parallel parameter. 

There are also operations between two files/layers. 
Examples are intersection, erase, union,... 

This is an example for the intersection operation:

.. code-block:: python

    from geofileops import geofileops

    geofileops.intersect(
            input1_path='...',
            input2_path='...',
            output_path='...')

Please check out all available operations in the API reference.

geofileops.geofile
------------------
This module contains some general purpose helper functions that might be of 
use when working with geofiles (or layers).
Remark: some functions will only work on Geopackage files, not on 
shapefiles.

Check out the API reference for more info.
