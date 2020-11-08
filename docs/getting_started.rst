
Getting started
===============

Installation
------------
GeofileOps is written in python, but it relies on several other libraries that 
have dependencies written in C. Those C dependencies can be difficult to 
install, but luckily the conda package management system also gives an easy 
alternative.

The easy way
--------
If you don't have miniconda or anaconda installed yet, here is a link to the 
`miniconda installer`_

Typically, the next step will be to create a new conda environment. The  
conda-forge channel is added here, as this channel contains the most recent 
versions of the dependencies::

    conda create -n geo
    conda activate geo
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict

Now the conda environment is prepared, you can install the GeofileOps 
dependencies::

    conda install --channel conda-forge python=3.8 geopandas=0.8 psutil

GeofileOps is not available as a conda package at the moment, but you can now 
install it using pip::

    pip install geofileops

The hard way
------------
If you are not able/willing to use conda to install the dependencies, please
follow the instructions to `install GeoPandas`_.

Afterwards you can install GeofileOps with pip.

.. miniconda installer_ : https://conda.io/projects/conda/en/latest/user-guide/install/index.html
.. install GeoPandas_: https://geopandas.org/install.html
