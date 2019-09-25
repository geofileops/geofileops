# geofile_ops
Library to make spatial operations on geo files fast and easy.

Remark: VERY early version, not feature complete and not production ready AT ALL!


Uses mainly ogr2ogr under the hood.

## Installation manual

1. Create and activate a new conda environment
```
conda create --name geofile_ops python=3.6
conda activate geofile_ops
```

2. Install the dependencies for the crop classification scripts:
```
conda install --channel conda-forge geopandas
```
Possibly you need to install your computer now, especially if it was the first time you installed anaconda/geopandas

3. Start the anaconda terminal window again and activate the environment
```
conda activate geofile_ops
```
