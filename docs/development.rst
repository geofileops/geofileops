
Development
===========

The source code can be found on the |geofileops git repository|.

If you want to do feature requests or file bug reports, that's the place to 
be as well.

Create development environment
------------------------------

The first step would be to fork the |geofileOps git repository|, to be able to run/debug 
the code.

If you don't have the conda package manager installed yet, here is a link to the 
`miniforge installer`_


Then you'll need to create a new conda environment with the necessary 
dependencies. ::

    conda env create -f environment-dev.yml
    conda activate geofileops-dev


Now you can install the pre-commit hook that will take care of some automatic checks
and formatting ::

    pre-commit install


.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3

.. |geofileOps git repository| raw:: html

   <a href="https://github.com/geofileops/geofileops" target="_blank">geofileops git repository</a>
