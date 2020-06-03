#-------------------------------------
# Import/init needed modules
#-------------------------------------
import logging
import os
from typing import Tuple

from osgeo import gdal

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\gdal"
    os.environ["PROJ_LIB"] = r"C:\Tools\miniconda3\envs\orthoseg\Library\share\proj"

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

class VectorTranslateInfo:
    def __init__(
            self,
            input_path: str, 
            output_path: str,
            translate_description: str = None,
            output_layer: str = None,
            spatial_filter: Tuple[float, float, float, float] = None,
            clip_bounds: Tuple[float, float, float, float] = None, 
            sqlite_stmt: str = None,
            transaction_size: int = 65536,
            append: bool = False,
            update: bool = False,
            create_spatial_index: bool = None,
            explodecollections: bool = False,
            force_output_geometrytype: str = None,
            priority_class: str = 'VERY_LOW',
            sqlite_journal_mode: str = 'WAL',
            verbose: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.translate_description = translate_description
        self.output_layer = output_layer
        self.spatial_filter = spatial_filter
        self.clip_bounds = clip_bounds
        self.sqlite_stmt = sqlite_stmt
        self.transaction_size = transaction_size
        self.append = append
        self.update = update
        self.create_spatial_index = create_spatial_index
        self.explodecollections = explodecollections
        self.force_output_geometrytype = force_output_geometrytype
        self.priority_class = priority_class
        self.sqlite_journal_mode = sqlite_journal_mode
        self.verbose = verbose
    
def vector_translate_by_info(info: VectorTranslateInfo):
        
    return vector_translate( 
            input_path=info.input_path,
            output_path=info.output_path,
            translate_description=info.translate_description,
            output_layer=info.output_layer,
            spatial_filter=info.spatial_filter,
            clip_bounds=info.clip_bounds,
            sqlite_stmt=info.sqlite_stmt,
            transaction_size=info.transaction_size,
            append=info.append,
            update=info.update,
            create_spatial_index=info.create_spatial_index,
            explodecollections=info.explodecollections,
            force_output_geometrytype=info.force_output_geometrytype,
            priority_class=info.priority_class,    
            sqlite_journal_mode=info.sqlite_journal_mode,
            verbose=info.verbose)

def vector_translate(
        input_path: str, 
        output_path: str,
        translate_description: str = None,
        output_layer: str = None,
        spatial_filter: Tuple[float, float, float, float] = None,
        clip_bounds: Tuple[float, float, float, float] = None, 
        sqlite_stmt: str = None,
        transaction_size: int = 65536,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = None,
        priority_class: str = 'VERY_LOW',
        sqlite_journal_mode: str = 'WAL',
        verbose: bool = False) -> bool:

    # Remark: when executing a select statement, I keep getting error that 
    # there are two columns named "geom" as he doesnt see the "geom" column  
    # in the select as a geometry column. Probably a version issue. Maybe 
    # try again later.

    args = []

    # Sql'ing, Filtering, clipping  
    if spatial_filter is not None:
        args.extend(['-spat', str(spatial_filter[0]), str(spatial_filter[1]), 
                    str(spatial_filter[2]), str(spatial_filter[3])])
    if clip_bounds is not None:
        args.extend(['-clipsrc', str(clip_bounds[0]), str(clip_bounds[1]), 
                    str(clip_bounds[2]), str(clip_bounds[3])])
    '''
    if sqlite_stmt is not None:
        args.extend(['-sql', sqlite_stmt, '-dialect', 'sqlite'])
    '''

    # Output file options
    if append is True:
        args.append('-append')
    if update is True:
        args.append('-update')

    # Files
    #args.append(output_path)
    #args.append(input_path)

    # Output layer options
    if explodecollections is True:
        args.append('-explodecollections')
    if output_layer is not None:
        args.extend(['-nln', output_layer])
    if force_output_geometrytype is not None:
        args.extend(['-nlt', force_output_geometrytype])
    if transaction_size is not None:
        args.extend(['-gt', str(transaction_size)])

    # Output layer creation options
    layerCreationOptions = []
    # TODO: should check if the layer exists instead of the file
    if not os.path.exists(output_path):
        if create_spatial_index is not None:
            if create_spatial_index is True:
                layerCreationOptions.extend(['SPATIAL_INDEX=YES'])
            else:
                layerCreationOptions.extend(['SPATIAL_INDEX=NO'])
    
    # Sqlite specific options
    datasetCreationOptions = []

    '''
    # Try if the busy_timeout isn't giving problems rather than solving them...
    if sqlite_journal_mode is not None:
        datasetCreationOptions.extend(['--config', 'OGR_SQLITE_PRAGMA', f"journal_mode={sqlite_journal_mode},busy_timeout=5000"])  
    else:
        datasetCreationOptions.extend(['--config OGR_SQLITE_PRAGMA busy_timeout=5000'])  
    '''
    if sqlite_journal_mode is not None:
        gdal.SetConfigOption('OGR_SQLITE_PRAGMA', f"journal_mode={sqlite_journal_mode}")

    #if append is False:
    #    args.extend(['--config', 'OGR_SQLITE_PRAGMA', '"journal_mode=WAL"'])
    #    args.extend(['-dsco', 'ADD_GPKG_OGR_CONTENTS=NO'])
    #else:
    #    args.extend(['--config', 'OGR_SQLITE_PRAGMA', 'busy_timeout=-1'])  
    #args.extend(['--config', 'OGR_SQLITE_SYNCHRONOUS', 'OFF'])  
    gdal.SetConfigOption('OGR_SQLITE_CACHE', '512')

    options = gdal.VectorTranslateOptions(
            options=args, 
            format=None, 
            accessMode=None, 
            srcSRS=None, 
            dstSRS=None, 
            reproject=True, 
            #SQLStatement=None, #sqlite_stmt,
            #SQLDialect=None, #'SQLITE',
            SQLStatement=sqlite_stmt,
            SQLDialect=None,
            where=None, 
            selectFields=None, 
            addFields=False, 
            forceNullable=False, 
            spatFilter=spatial_filter, 
            spatSRS=None,
            datasetCreationOptions=datasetCreationOptions, 
            layerCreationOptions=layerCreationOptions, 
            layers=None, # TODO: implement! [output_layer]
            layerName=output_layer,
            geometryType=None, 
            dim=None, 
            segmentizeMaxDist=None, 
            zField=None, 
            skipFailures=False, 
            limit=None, 
            callback=None, 
            callback_data=None)

    input_ds = None
    try: 
        # In some cases gdal only raises the last exception instead of the stack in VectorTranslate, 
        # so you lose necessary details! -> uncomment gdal.DontUseExceptions() when debugging!
        #gdal.DontUseExceptions()
        if verbose:
            logger.info(f"Execute {sqlite_stmt} on {input_path}")
        input_ds = gdal.OpenEx(input_path)

        # TODO: memory output support might be interesting to support
        ret_val = gdal.VectorTranslate(
                destNameOrDestDS=output_path,
                srcDS=input_ds,
                options=options)
        if ret_val is None:
            raise Exception("BOEM")
    except Exception as ex:
        message = f"Error executing {sqlite_stmt}"
        logger.exception(message)
        raise Exception(message) from ex
    finally:
        if input_ds is not None:
            del input_ds
        
    return True

def vector_translate_async(
        concurrent_pool,
        info: VectorTranslateInfo) -> bool:

    #return vector_translate_by_info(info)
    return concurrent_pool.submit(
            vector_translate_by_info,
            info)

def vector_info(
        path: str, 
        task_description = None,
        layer: str = None,
        readonly: bool = False,
        report_summary: bool = False,
        sqlite_stmt: str = None,        
        verbose: bool = False):
    """"Run a command"""

    raise Exception("Not implemented")

if __name__ == '__main__':
    None
