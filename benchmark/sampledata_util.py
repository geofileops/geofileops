# -*- coding: utf-8 -*-
"""
Module containing some utilities to get sample data.
"""

import enum
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Optional
import urllib.request
import zipfile

from geofileops import geofileops

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

class SampleGeofile(enum.Enum):
    POLYGON_AGRI_PARCEL_2018 = enum.auto()
    POLYGON_AGRI_PARCEL_2019 = enum.auto()

    @property
    def url(self):
        if self.name == 'POLYGON_AGRI_PARCEL_2018':
            return r"https://downloadagiv.blob.core.windows.net/landbouwgebruikspercelen/2018/Landbouwgebruikspercelen_LV_2018_GewVLA_Shape.zip"
        elif self.name == 'POLYGON_AGRI_PARCEL_2019':
            return r"https://downloadagiv.blob.core.windows.net/landbouwgebruikspercelen/2019/Landbouwgebruikspercelen_LV_2019_GewVLA_Shapefile.zip"

    @property
    def defaultname(self):
        return f"{self.name.lower()}.gpkg"

    @property
    def defaultpath(self):
        return Path(tempfile.gettempdir()) / 'geofileops_sampledata' / self.defaultname

    def custompath(
            self, 
            base_dir: Optional[Path] = None):
        if base_dir is None:
            return self.defaultpath
        else:
            return base_dir / 'geofileops_sampledata' / self.defaultname
        
def download_samplefile(
        samplegeofile: SampleGeofile,
        dst_path: Optional[Path] = None) -> Path:
    """
    Download a sample file to dest_path.

    Args:
        samplegeofile (SampleGeofile): the sample file to retrieve
        dst_path (Path): the location to downloaded the sample file to. 
            If it is a directory, samplegeofile.defaultname will be used for
            the file name. If it is None, samplegeofile.defaultpath will be 
            used. Defaults to None.

    Returns:
        Path: the path to the downloaded sample file.
    """

    # If the destination path is a directory, use the default file name
    if dst_path is None or (dst_path.is_dir() or (dst_path.exists() is False and dst_path.suffix == '')):
        dst_path = samplegeofile.custompath(dst_path)
    # If the sample file already exists, return
    if dst_path.exists():
        return dst_path
    # Make sure the destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Download zip file if needed...  
    zip_path = dst_path.parent / f"{dst_path.stem}.zip"
    unzippedzip_dir = dst_path.parent / zip_path.stem
    if not zip_path.exists() and not unzippedzip_dir.exists():
        # Download beschmark file
        logger.info(f"Download sample data to {dst_path}")
        urllib.request.urlretrieve(str(samplegeofile.url), zip_path)
    
    # Unzip zip file if needed... 
    if not unzippedzip_dir.exists():
        # Unzip file
        logger.info('Unzip sample data')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzippedzip_dir)
            
    # Convert shapefile to geopackage + make valid
    shp_dir = unzippedzip_dir / 'Shapefile'
    shp_paths = list(shp_dir.glob('Lbgbrprc*.shp'))
    if len(shp_paths) != 1:
        raise Exception(f"Should find 1 shapefile, found {len(shp_paths)}")

    logger.info(f"Make shapefile valid and save to {dst_path}")
    geofileops.makevalid(shp_paths[0], dst_path)
    
    # Cleanup
    if zip_path.exists():
        zip_path.unlink()
    if unzippedzip_dir.exists():
        shutil.rmtree(unzippedzip_dir)
    
    return dst_path
