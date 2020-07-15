import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geofileops", 
    version="0.0.5",
    author="Pieter Roggemans",
    author_email="pieter.roggemans@gmail.com",
    description="Package to do spatial operations on geo files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theroggy/geofileops",
    packages=setuptools.find_packages(),
    install_requires=["geopandas>=0.8", "pygeos", "psutil"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)