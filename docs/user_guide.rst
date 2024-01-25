.. currentmodule:: geofileops

==========
User guide
==========

The main objective of geofileops is to provide a simple to use but powerful API to do
fast spatial operations on large vector GIS files.

General  
-------

Because geofileops uses multiprocessing under the hood, it is a good idea to always use
the ``if __name__ == "__main__":`` block in standalone Python scripts to avoid errors.
More information in :ref:`FAQ - Standalone scripts<FAQ-standalone-scripts>`

Also interesting to know: because processing large files can take some time, geofileops
logs progress info.

Combining both, a basic standalone script using geofileops can looks like this:

.. code-block:: python

    import logging
    import geofileops as gfo
    
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)
        gfo.buffer(input_path="input.gpkg", output_path="output.gpkg", distance=2)


Geometry tools  
--------------

The typical geometry operations are directly supported, eg. :meth:`~buffer`,
:meth:`~simplify`, :meth:`~convexhull`, ...

.. code-block:: python

    gfo.simplify(input_path="...", output_path="...", tolerance=1)


For more advanced uses, you can execute any sqlite sql statement on an input file using
:meth:`~select`. Because |spatialite_reference_link| functions are also supported, this
is quite powerful. To simplify the sql statements, there are some placeholders you can
use that will be filled out by geofileops:

.. code-block:: python

    city = "Brussels"
    sql_stmt = f"""
        SELECT ST_OrientedEnvelope({{geometrycolumn}}) AS geom
              {{columns_to_select_str}}
          FROM "{{input_layer}}" layer
         WHERE city_name = {city}"
    """
    gfo.select(
        input_path="...",
        output_path="...",
        columns=["city_name", "city_code"],
        sql_stmt=sql_stmt,
    )

Finally, you can apply any python function on the geometry column using :meth:`~apply`.

.. code-block:: python

    import pygeoops

    def cleanup(geom, min_area_to_keep):
        new_geom = pygeoops.remove_inner_rings(geom, min_area_to_keep=min_area_to_keep)
        return new_geom

    gfo.apply(
        input_path="...",
        output_path="...",
        func=lambda geom: cleanup(geom, min_area_to_keep=1),
    )

Most spatial operations in geofileops have the same optional parameters. These are the
most interesting ones:

* **columns**: if not specified, all standard attribute columns will be retained in the
  output file. If you don't need all columns, specify the ones you want to keep. 
  You can retain a copy of the special "fid" column in the output file by specifing
  "fid" in addition to the standard attribute columns you want to retain.
* **explodecollections**: the output features will be "exploded", so multipart
  features will be converted to single parts.
* **gridsize**: the size of the grid the coordinates of the ouput will be rounded to.
  Eg. 0.001 to keep 3 decimals. If eg. a polygon is narrower than the ``gridsize``, it
  will be removed. Value 0.0, the default, doesn't change the precision.
* **force**: by default, if the ``output_path`` already exists, geofileops will just log
  this and return. To overwrite the existing ``output_path``, specify ``force=True``.
    
Spatial operations between two files/layers
-------------------------------------------

For operations between two layers, obviously 2 input files/layers need to be 
specified.

The standard spatial operations are supported, eg. :meth:`~intersection`, 
:meth:`~erase`, :meth:`~union`,...

More specific features are:

* :meth:`~select_two_layers`: execute a select statement (including  
  |spatialite_reference_link|) on the input files. 
* :meth:`~join_nearest`: find the n nearest features from one layer 
  compared the other.

This is a code example for the intersection operation:

.. code-block:: python

    gfo.intersection(input1_path="...", input2_path="...", output_path="...")


The two-layer operations will have about the same optional parameters as the single
layer operations, but where applicable they are duplicated for input1 and input2.


General file/layer operations
-----------------------------

Finally there are also some functions available to manipulate geo files or 
layers. Eg. :meth:`~copy`, :meth:`~move`, :meth:`~get_layerinfo`, 
:meth:`~add_column`,...

This is an example to get information about the (only) layer in a geo file: 

.. code-block:: python

    layerinfo = gfo.get_layerinfo(path='...')
    print(f"Layer {layerinfo.name} contains {layerinfo.featurecount} features")

Remark: some functions might only work on Geopackage files, not on shapefiles.

.. |spatialite_reference_link| raw:: html

   <a href="https://www.gaia-gis.it/gaia-sins/spatialite-sql-latest.html" target="_blank">spatialite</a>
