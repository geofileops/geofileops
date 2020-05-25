import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geofile_ops", 
    version="0.0.4",
    author="Pieter Roggemans",
    author_email="pieter.roggemans@gmail.com",
    description="Package to do spatial operations on data in a geofile.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theroggy/geofile_ops",
    packages=setuptools.find_packages(),
    install_requires=["geopandas>=0.7"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)