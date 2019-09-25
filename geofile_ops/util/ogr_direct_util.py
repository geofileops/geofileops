#-------------------------------------
# Import/init needed modules
#-------------------------------------
import logging
import os
import datetime
import errno
import subprocess
import concurrent.futures
#import _winreg as winreg

from osgeo import gdal
from osgeo import ogr

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def vector_translate(
        input_filepath: str, 
        output_filepath: str,
        output_layer: str = None,
        spatial_filter: () = None,
        clip_bounds: () = None, 
        sqlite_stmt: str = None,
        transaction_size: int = 65536,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = None,
        verbose: bool = False):

    # Remark: when executing a select statement, I keep getting error that 
    # there are two columns named "geom" as he doesnt see the "geom" column  
    # in the select as a geometry column. Probably a version issue. Maybe 
    # try again later.

    options = gdal.VectorTranslateOptions(
            SQLDialect='SQLITE',
            SQLStatement=sqlite_stmt,
            layerName=output_layer)
            #,geometryType='MULTIPOLYGON')

    try: 
        # In some cases gdal only raises the last exception instead of the stack in VectorTranslate, 
        # so you lose necessary details! -> uncomment gdal.DontUseExceptions() when debugging!
        #gdal.DontUseExceptions()
        logger.info(f"Execute {sqlite_stmt} on {input_filepath}")
        input_ds = gdal.OpenEx(input_filepath)

        # TODO: memory output support might be interesting to support
        ret_val = gdal.VectorTranslate(
                destNameOrDestDS=output_filepath,
                srcDS=input_ds,
                options=options)
        if ret_val is None:
            raise Exception("BOEM")
    except Exception as ex:
        message = f"Error executing {sqlite_stmt}"
        logger.exception(message)
        raise Exception(message) from ex
        
    return 'OK'

def vector_translate_async(
        concurrent_pool,
        input_filepath: str, 
        output_filepath: str,
        output_layer: str = None,
        spatial_filter: () = None,
        clip_bounds: () = None, 
        sqlite_stmt: str = None,
        transaction_size: int = 65536,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = None,
        verbose: bool = False):

    return concurrent_pool.submit(
            execute,
            input_filepath, 
            output_filepath,
            output_layer,
            spatial_filter,
            clip_bounds, 
            sqlite_stmt,
            transaction_size,
            append,
            update,
            create_spatial_index,
            explodecollections,
            force_output_geometrytype,
            verbose)

def vector_info(
        filepath: str, 
        layer: str = None,
        readonly: bool = False,
        report_summary: bool = False,
        sqlite_stmt: str = None,        
        verbose: bool = False):
    """"Run a command"""

    raise Exception("Not implemented")

if __name__ == '__main__':
    None
        