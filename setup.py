"""
Setup file to package geofileops.
"""

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("geofileops/version.txt") as file:
    version = file.readline()

setuptools.setup(
    name="geofileops",
    version=version,
    author="Pieter Roggemans",
    author_email="pieter.roggemans@gmail.com",
    description="Python toolbox to process large vector files faster.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geofileops/geofileops",
    include_package_data=True,
    packages=setuptools.find_packages(
        exclude=("tests", "benchmark", "benchmark.*"),
        include=("geofileops", "geofileops.*"),
    ),
    install_requires=[
        "cloudpickle",
        "gdal>=3.6,<3.10",
        "geopandas>=0.12,<1",
        "numpy",
        "packaging",
        "pandas",
        "psutil",
        "pygeoops>=0.4,<0.5",
        "pyogrio",
        "pyproj",
        "shapely>=2,<2.1",
    ],
    extras_require={"full": ["simplification"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.9",
)
