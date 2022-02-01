
Getting started
===============

Installation
------------
geofileops is written in python, but it relies on several other libraries that 
have dependencies written in C. Those C dependencies can be difficult to 
install, but luckily the conda package management system also gives an easy 
alternative.

The easy way
------------
If you don't have miniconda or anaconda installed yet, here is a link to the 
`miniconda installer`_

Typically, the next step will be to create a new conda environment. The  
conda-forge channel is added here, as this channel contains the most recent 
versions of the dependencies::

    conda create -n geo
    conda activate geo
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict

Now the conda environment is prepared, you can install the geofileops 
dependencies that are available as conda packages::

    conda install --channel conda-forge python=3.9 geofileops

The hard way
------------
If you are not able/willing to use conda to install the dependencies, please
follow the instructions to `install GeoPandas`_.

Afterwards you can install geofileops with pip.

.. _miniconda installer : https://conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _install GeoPandas : https://geopandas.org/install.html
