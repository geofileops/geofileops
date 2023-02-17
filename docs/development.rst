
Development
===========

The source code can be found on the |geofileops git repository|.

If you want to do feature requests or file bug reports, that's the place to 
be as well.

Create development environment
------------------------------

The first step would be Now, if you fork the |geofileOps git repository|, you should be able to run/debug the code.

If you don't have the conda package manager installed yet, here is a link to the 
`miniforge installer`_


Then you'll need to create a new conda environment with the necessary 
dependencies::

    conda create -n geofileopsdev
    conda activate geofileopsdev
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    conda install python=3.9 geopandas=0.10 psutil
    conda install pylint pytest rope pydata-sphinx-theme sphinx-automodapi



.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3

.. |geofileOps git repository| raw:: html

   <a href="https://github.com/geofileops/geofileops" target="_blank">geofileops git repository</a>
