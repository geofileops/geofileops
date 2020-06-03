
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

def select(
        input_path: str,
        output_path: str,
        sqlite_stmt: str,
        input_layer: str = None,        
        output_layer: str = None,
        verbose: bool = False,
        force: bool = False):

    return geofileops_ogr.select(
            input_path=input_path,
            output_path=output_path,
            sqlite_stmt=sqlite_stmt,
            input_layer=input_layer,        
            output_layer=output_layer,
            verbose=verbose,
            force=force)

def convexhull(
        input_path: str,
        output_path: str,
        input_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    return geofileops_ogr.convexhull(
            input_path=input_path,
            output_path=output_path,
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def buffer(
        input_path: str,
        output_path: str,
        buffer: float,
        quadrantsegments: int = 5,
        input_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    return geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            buffer=buffer,
            quadrantsegments=quadrantsegments,
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def simplify(
        input_path: str,
        output_path: str,
        tolerance: float,        
        input_layer: str = None,        
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    return geofileops_ogr.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=tolerance,
            input_layer=input_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def intersect(
        input1_path: str,
        input2_path: str,
        output_path: str,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):
    """
    Calculate the pairwise intersection of alle features in input1 with all 
    features in input2.
    
    Args:
        input1_path (str): [description]
        input2_path (str): [description]
        output_path (str): [description]
        input1_layer (str, optional): [description]. Defaults to None.
        input2_layer (str, optional): [description]. Defaults to None.
        output_layer (str, optional): [description]. Defaults to None.
        nb_parallel (int, optional): [description]. Defaults to -1.
        force (bool, optional): [description]. Defaults to False.
    
    Raises:
        Exception: [description]
    """

    return geofileops_ogr.intersect(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            input1_layer=input1_layer,
            input2_layer=input2_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def export_by_location(
        input_to_select_from_path: str,
        input_to_compare_with_path: str,
        output_path: str,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    return geofileops_ogr.export_by_location(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path,
            input1_layer=input1_layer,
            input2_layer=input2_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def export_by_distance(
        input_to_select_from_path: str,
        input_to_compare_with_path: str,
        output_path: str,
        max_distance: float,
        input1_layer: str = None,
        input2_layer: str = None,
        output_layer: str = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    return geofileops_ogr.export_by_distance(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path,
            max_distance=max_distance,
            input1_layer=input1_layer,
            input2_layer=input2_layer,
            output_layer=output_layer,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)

def dissolve(
        input_path: Union[str, 'os.PathLike[Any]'],  
        output_path: Union[str, 'os.PathLike[Any]'],
        groupby_columns: Optional[List[str]] = None,
        aggfunc: str = None,
        explodecollections: bool = False,
        keep_cardsheets: bool = False,
        input_layer: str = None,        
        output_layer: str = None,
        input_cardsheets_path: Union[str, 'os.PathLike[Any]'] = None,
        nb_parallel: int = -1,
        verbose: bool = False,
        force: bool = False):

    return geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=groupby_columns,
            aggfunc=aggfunc,
            explodecollections=explodecollections,
            keep_cardsheets=keep_cardsheets,
            input_layer=input_layer,        
            output_layer=output_layer,
            input_cardsheets_path=input_cardsheets_path,
            nb_parallel=nb_parallel,
            verbose=verbose,
            force=force)
    
if __name__ == '__main__':
    raise Exception("Not implemented!")
