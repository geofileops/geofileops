# -*- coding: utf-8 -*-
"""
Module exposing all supported operations on geomatries in geofiles.
"""

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, AnyStr, List, Optional, Tuple, Union

from .util import geofileops_ogr
from .util import geofileops_gpd

################################################################################
# Some init
################################################################################

logger = logging.getLogger(__name__)

################################################################################
# The real work
################################################################################

def buffer(
        input_path: Union[str, 'os.PathLike[Any]'],
        output_path: Union[str, 'os.PathLike[Any]'],
        distance: float,
        quadrantsegments: int = 5,
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Applies a buffer operation on geometry column of the input file.
    
    The result is written to the output file specified. 

    Args:
        input_path (PathLike): the input file
        output_path (PathLike): the file to write the result to
        distance (float): the buffer size to apply
        quadrantsegments (int): the number of points an arc needs to be 
                approximated with. Defaults to 5.
        input_layer (str, optional): input layer name. Optional if the input 
                file only contains one layer.
        output_layer (str, optional): input layer name. Optional if the input 
                file only contains one layer.
        columns (List[str], optional): list of columns to return. If None,
                all columns are returned.
        nb_parallel (int, optional): the number of parallel processes to use. 
                If not specified, all available processors will be used.
        verbose (bool, optional): write more info to the output. 
                Defaults to False.
        force (bool, optional): overwrite existing output file(s). 
                Defaults to False.
    """
    logger.info(f"Start buffer on {input_path} with distance: {distance} and quadrantsegments: {quadrantsegments}")
    return geofileops_gpd.buffer(
            input_path=Path(input_path),
            output_path=Path(output_path),
            distance=distance,
            quadrantsegments=quadrantsegments,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def convexhull(
        input_path: Union[str, 'os.PathLike[Any]'],
        output_path: Union[str, 'os.PathLike[Any]'],
        input_layer: str = None,
        output_layer: str = None,
        columns: List[str] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Applies a convexhull operation on the input file.
    
    The result is written to the output file specified. 

    Args:
        input_path (PathLike): the input file
        output_path (PathLike): the file to write the result to
        input_layer (str, optional): input layer name. Optional if the input 
            file only contains one layer.
        output_layer (str, optional): input layer name. Optional if the input 
            file only contains one layer.
        nb_parallel (int, optional): the number of parallel processes to use. 
            If not specified, all available processors will be used.
        columns (List[str], optional): If not None, only output the columns 
            specified. Defaults to None.
        verbose (bool, optional): write more info to the output. 
            Defaults to False.
        force (bool, optional): overwrite existing output file(s). 
            Defaults to False.
    """
    logger.info(f"Start convexhull on {input_path}")
    return geofileops_gpd.convexhull(
            input_path=Path(input_path),
            output_path=Path(output_path),
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def dissolve(
        input_path: Union[str, 'os.PathLike[Any]'],  
        output_path: Union[str, 'os.PathLike[Any]'],
        groupby_columns: Optional[List[str]] = None,
        aggfunc: str = 'first',
        explodecollections: bool = False,
        clip_on_tiles: bool = False,
        tiles_path: Union[str, 'os.PathLike[Any]'] = None,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Applies a dissolve operation on geometry column of the input file.
    
    The result is written to the output file specified. 

    Args:
        input_path (PathLike): the input file
        output_path (PathLike): the file to write the result to
        groupby_columns: (List[str]): list of columns to group on before applying the union.
        explodecollections (bool, optional): True to convert all multi-geometries to 
                singular ones after the dissolve. Defaults to False.
        clip_on_tiles (bool, optional): True to clip the result on the tiles used.
        input_layer (str, optional): input layer name. Optional if the  
                file only contains one layer.
        output_layer (str, optional): input layer name. Optional if the  
                file only contains one layer.
        nb_parallel (int, optional): the number of parallel processes to use. 
                If not specified, all available processors will be used.
        verbose (bool, optional): write more info to the output. 
                Defaults to False.
        force (bool, optional): overwrite existing output file(s). 
                Defaults to False.
    """
    # Init
    tiles_path_p = None
    if tiles_path is not None:
        tiles_path_p = Path(tiles_path)

    logger.info(f"Start dissolve on {input_path} to {output_path}")
    return geofileops_gpd.dissolve(
            input_path=Path(input_path),
            output_path=Path(output_path),
            groupby_columns=groupby_columns,
            aggfunc=aggfunc,
            explodecollections=explodecollections,
            clip_on_tiles=clip_on_tiles,
            input_layer=input_layer,        
            output_layer=output_layer,
            tiles_path=tiles_path_p,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def isvalid(
        input_path: Union[str, 'os.PathLike[Any]'],
        output_path: Union[str, 'os.PathLike[Any]'] = None,
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False) -> bool:

    # Check parameters
    if output_path is not None:
        output_path_p = Path(output_path)
    else:
        input_path_p = Path(input_path)
        output_path_p = input_path_p.parent / f"{input_path_p.stem}_isvalid{input_path_p.suffix}" 

    # Go!
    logger.info(f"Start isvalid on {input_path}")
    return geofileops_ogr.isvalid(
            input_path=Path(input_path),
            output_path=output_path_p,
            input_layer=input_layer,        
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def makevalid(
        input_path: Union[str, 'os.PathLike[Any]'],
        output_path: Union[str, 'os.PathLike[Any]'],
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    logger.info(f"Start makevalid on {input_path}")
    return geofileops_ogr.makevalid(
            input_path=Path(input_path),
            output_path=Path(output_path),
            input_layer=input_layer,        
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def select(
        input_path: Union[str, 'os.PathLike[Any]'],
        output_path: Union[str, 'os.PathLike[Any]'],
        sql_stmt: str,
        sql_dialect: str = 'SQLITE',
        input_layer: str = None,        
        output_layer: str = None,
        columns: List[str] = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = None,
        nb_parallel: int = 1,
        verbose: bool = False,
        force: bool = False):
    """
    Execute an sqlite style SQL query on the file. 
    
    By convention, the sqlite query can contain following placeholders that
    will be automatically replaced for you:
      * {geometrycolumn}: the column where the primary geometry is stored.
      * {columns_to_select_str}: if 'columns' is not None, those columns, 
        otherwise all columns of the layer.
      * {input_layer}: the layer name of the input layer.
      * {batch_filter}: the filter use to process in parallel per batch. 
    
    Because some sql statement won't give the same result when parallellized 
    (eg. when using a group by statement), nb_parallel is 1 by default. 
    If you do want to use parallel processing, 
    specify nb_parallel + make sure to include the placeholder {batch_filter} 
    in your sql_stmt. This placeholder will be replaced with a filter
    of the form 'AND rowid >= x AND rowid < y'.

    The result is written to the output file specified.

    Args:
        input_path (PathLike): the input file
        output_path (PathLike): the file to write the result to
        sql_stmt (str): the statement to execute
        sql_dialect (str, optional): the sql dialect to use. If None is passed,
            the default sql dialect of the underlying source is used. The 
            default is 'SQLITE'.
        input_layer (str, optional): input layer name. Optional if the input 
            file only contains one layer.
        output_layer (str, optional): input layer name. Optional if the input
            file only contains one layer.
        columns (List[str], optional): If not None AND the column placeholders 
            are used in the sql statement, only output the columns specified. 
            Defaults to None.
        explodecollections (bool, optional): True to convert all multi-geometries to 
            singular ones after the dissolve. Defaults to False.
        force_output_geometrytype (str, optional): The output geometry type to 
            force. Defaults to None, and then the geometry type of the input is used 
        nb_parallel (int, optional): the number of parallel processes to use. 
            Defaults to 1. To use all available cores, pass -1. 
        verbose (bool, optional): write more info to the output. Defaults to False.
        force (bool, optional): overwrite existing output file(s). Defaults to False.
    """
    logger.info(f"Start select {sql_stmt} \n    on {input_path}")
    return geofileops_ogr.select(
            input_path=Path(input_path),
            output_path=Path(output_path),
            sql_stmt=sql_stmt,
            sql_dialect=sql_dialect,
            input_layer=input_layer,        
            output_layer=output_layer,
            columns=columns,
            explodecollections=explodecollections,
            force_output_geometrytype=force_output_geometrytype,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def simplify(
        input_path: Union[str, 'os.PathLike[Any]'],
        output_path: Union[str, 'os.PathLike[Any]'],
        tolerance: float,        
        input_layer: str = None,        
        output_layer: str = None,
        columns: List[str] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Applies a simplify operation on geometry column of the input file.
    
    The result is written to the output file specified. 

    Args:
        input_path (PathLike): the input file
        output_path (PathLike): the file to write the result to
        tolerance (float): the tolerancy to use when simplifying
        input_layer (str, optional): input layer name. Optional if the input 
            file only contains one layer.
        output_layer (str, optional): input layer name. Optional if the input 
            file only contains one layer.
        columns (List[str], optional): If not None, only output the columns 
            specified. Defaults to None.
        nb_parallel (int, optional): the number of parallel processes to use. 
            If not specified, all available processors will be used.
        verbose (bool, optional): write more info to the output. 
            Defaults to False.
        force (bool, optional): overwrite existing output file(s). 
            Defaults to False.
    """
    logger.info(f"Start simplify on {input_path} with tolerance {tolerance}")
    return geofileops_gpd.simplify(
            input_path=Path(input_path),
            output_path=Path(output_path),
            tolerance=tolerance,
            input_layer=input_layer,
            output_layer=output_layer,
            columns=columns,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

if __name__ == '__main__':
    raise Exception("Not implemented!")
