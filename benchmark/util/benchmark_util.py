# -*- coding: utf-8 -*-
"""
Module containing some utilities to get sample data.
"""

import datetime
import logging
from pathlib import Path
import pprint
import shutil
import tempfile
from typing import Optional
import urllib.request
import zipfile

import geofileops as gfo

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

class BenchmarkResult:
    def __init__(self, 
            package: str,
            version: str,
            operation: str,
            secs_taken: float,
            run_details: dict):
        self.datetime = datetime.datetime.now()
        self.package = package
        self.version = version
        self.operation = operation
        self.secs_taken = secs_taken
        self.run_details = run_details
        
    def __repr__(self):
        return f"{self.__class__}({self.__dict__})"

def prepare_dst_path(
        dst_name: str,
        dst_dir: Optional[Path] = None):
    if dst_dir is None:
        return Path(tempfile.gettempdir()) / 'geofileops_sampledata' / dst_name
    else:
        return dst_dir / dst_name
        
def download_samplefile(
        url: str,
        dst_name: str,
        dst_dir: Optional[Path] = None) -> Path:
    """
    Download a sample file to dest_path.

    If it is zipped, it will be unzipped. If needed, it will be converted to 
    the file type as determined by the suffix of dst_name.

    Args:
        url (str): the url of the file to download
        dst_dir (Path): the dir to downloaded the sample file to. 
            If it is None, a dir in the default tmp location will be 
            used. Defaults to None.

    Returns:
        Path: the path to the downloaded sample file.
    """
    # If the destination path is a directory, use the default file name
    dst_path = prepare_dst_path(dst_name, dst_dir)
    # If the sample file already exists, return
    if dst_path.exists():
        return dst_path
    # Make sure the destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # If the url points to a file with the same suffix as the dst_path, 
    # just download
    url_path = Path(url) 
    if url_path.suffix.lower() == dst_path.suffix.lower():
        logger.info(f"Download to {dst_path}")
        urllib.request.urlretrieve(url, dst_path)
    else:
        # The file downloaded is different that the destination wanted, so some 
        # converting will need to be done
        tmp_dir = dst_path.parent / "tmp"
        
        try:
            # Remove tmp dir if it exists already
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
        
            # Download file
            tmp_path = tmp_dir / f"{dst_path.stem}{url_path.suffix.lower()}"
            logger.info(f"Download tmp data to {tmp_path}")
            urllib.request.urlretrieve(url, tmp_path)
            
            # If the temp file is a .zip file, unzip to dir
            if tmp_path.suffix == ".zip":
                # Unzip
                unzippedzip_dir = dst_path.parent / tmp_path.stem
                logger.info(f"Unzip to {unzippedzip_dir}")
                with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                    zip_ref.extractall(unzippedzip_dir)
                
                # Look for the file
                tmp_paths = []
                for suffix in [".shp", ".gpkg"]:
                    tmp_paths.extend(list(unzippedzip_dir.rglob(f"*{suffix}")))
                if len(tmp_paths) == 1:
                    tmp_path = tmp_paths[0]
                else:
                    raise Exception(f"Should find 1 geofile, found {len(tmp_paths)}: \n{pprint.pformat(tmp_paths)}")

            if dst_path.suffix == tmp_path.suffix:
                gfo.move(tmp_path, dst_path)
            else:
                logger.info(f"Convert tmp file to {dst_path}")
                gfo.makevalid(tmp_path, dst_path)
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
    
    return dst_path
