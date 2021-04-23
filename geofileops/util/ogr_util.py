# -*- coding: utf-8 -*-
"""
Module containing utilities regarding the usage of ogr functionalities.
"""

#-------------------------------------
# Import/init needed modules
#-------------------------------------
from concurrent import futures
from datetime import datetime
import logging
import multiprocessing
import os
from pathlib import Path
import pprint
import subprocess
import tempfile
from threading import Lock
import time
from typing import Any, List, Optional, Tuple, Union
from io import StringIO
import re

import geopandas as gpd
from osgeo import gdal
gdal.UseExceptions() 

from geofileops import geofile
from geofileops.util import general_util
from geofileops.util.general_util import MissingRuntimeDependencyError
from geofileops.util.geometry_util import GeometryType

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Make sure only one instance per process is running
lock = Lock()

# Get a logger...
logger = logging.getLogger(__name__)

# Cache the result of the check because it is quite expensive
global_spatiallite_versioninfo_default = None
global_spatiallite_versioninfo_alternative = None

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

class VectorTranslateInfo:
    def __init__(
            self,
            input_path: Path, 
            output_path: Path,
            translate_description: str = None,
            input_layers: Union[Optional[List[str]], str] = None,
            output_layer: str = None,
            spatial_filter: Tuple[float, float, float, float] = None,
            clip_bounds: Tuple[float, float, float, float] = None, 
            sql_stmt: str = None,
            sql_dialect: str = None,
            transaction_size: int = 65536,
            append: bool = False,
            update: bool = False,
            create_spatial_index: bool = None,
            explodecollections: bool = False,
            force_output_geometrytype: GeometryType = None,
            priority_class: str = 'VERY_LOW',
            sqlite_journal_mode: str = None,
            force_py: bool = False,
            verbose: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.translate_description = translate_description
        self.input_layers = input_layers
        self.output_layer = output_layer
        self.spatial_filter = spatial_filter
        self.clip_bounds = clip_bounds
        self.sql_stmt = sql_stmt
        self.sql_dialect = sql_dialect
        self.transaction_size = transaction_size
        self.append = append
        self.update = update
        self.create_spatial_index = create_spatial_index
        self.explodecollections = explodecollections
        self.force_output_geometrytype = force_output_geometrytype
        self.priority_class = priority_class
        self.sqlite_journal_mode = sqlite_journal_mode
        self.force_py = force_py
        self.verbose = verbose

def vector_translate_by_info(info: VectorTranslateInfo):

    return vector_translate( 
            input_path=info.input_path,
            output_path=info.output_path,
            translate_description=info.translate_description,
            input_layers=info.input_layers,
            output_layer=info.output_layer,
            spatial_filter=info.spatial_filter,
            clip_bounds=info.clip_bounds,
            sql_stmt=info.sql_stmt,
            sql_dialect=info.sql_dialect,
            transaction_size=info.transaction_size,
            append=info.append,
            update=info.update,
            create_spatial_index=info.create_spatial_index,
            explodecollections=info.explodecollections,
            force_output_geometrytype=info.force_output_geometrytype,
            priority_class=info.priority_class,    
            sqlite_journal_mode=info.sqlite_journal_mode,
            force_py=info.force_py,
            verbose=info.verbose)

def vector_translate_async(
        concurrent_pool,
        info: VectorTranslateInfo):

    return concurrent_pool.submit(
            vector_translate_by_info,
            info)

def vector_translate_seq(
        vector_translate_infos: List[VectorTranslateInfo]):

    for vector_translate_info in vector_translate_infos:
        vector_translate_by_info(vector_translate_info)

def vector_translate_parallel(
        vector_translate_infos: List[VectorTranslateInfo],
        operation_description: str,
        nb_parallel: int = -1):

    # Init
    if len(vector_translate_infos) == 0:
        return
    start_time = datetime.now()
    if nb_parallel == -1:
        nb_parallel = multiprocessing.cpu_count()
    
    with futures.ThreadPoolExecutor(nb_parallel) as concurrent_pool:
        
        future_to_calc_id = {}
        for calc_id, vector_translate_info in enumerate(vector_translate_infos):
            future = vector_translate_async(
                    concurrent_pool=concurrent_pool,
                    info=vector_translate_info)
            future_to_calc_id[future] = calc_id
        
        # Wait till all parallel processes are ready
        nb_done = 0
        for future in futures.as_completed(future_to_calc_id):
            calc_id = future_to_calc_id[future]
            try:
                _ = future.result()
            except Exception as ex:
                message = f"Async translate {calc_id} ERROR: {vector_translate_infos[calc_id].translate_description}\n{pprint.pformat(vector_translate_infos[calc_id])}"
                raise Exception(message) from ex

            # Log the progress and prediction speed
            nb_done += 1
            general_util.report_progress(start_time, nb_done, len(vector_translate_infos), operation_description)      

def vector_translate(
        input_path: Path, 
        output_path: Path,
        translate_description: str = None,
        input_layers: Union[Optional[List[str]], str] = None,
        output_layer: str = None,
        spatial_filter: Tuple[float, float, float, float] = None,
        clip_bounds: Tuple[float, float, float, float] = None, 
        sql_stmt: str = None,
        sql_dialect: str = None,
        transaction_size: int = 65536,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = None,
        explodecollections: bool = False,
        force_output_geometrytype: GeometryType = None,
        priority_class: str = 'VERY_LOW',
        sqlite_journal_mode: str = None,
        force_py: bool = False,
        verbose: bool = False) -> bool:
    """
    Run a command
    """
    # If running a sqlite statement on a gpkg file, check first if spatialite works properly
    # Remark: if force_py is True, the user should know what he is doing 
    sql_dialect_to_use = sql_dialect
    if force_py:
        gdal_to_use = 'gdal_default'
    elif(sql_stmt is not None 
       and sql_dialect is not None
       and sql_dialect.upper() == 'SQLITE' 
       and input_path.suffix.lower() == '.gpkg'):
        gdal_to_use, sql_dialect_to_use = get_gdal_to_use(sql_stmt)
    else:
        gdal_to_use = 'gdal_default'

    if gdal_to_use == 'gdal_default':
        return vector_translate_py( 
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            input_layers=input_layers,
            output_layer=output_layer,
            spatial_filter=spatial_filter,
            clip_bounds=clip_bounds,
            sql_stmt=sql_stmt,
            sql_dialect=sql_dialect_to_use,
            transaction_size=transaction_size,
            append=append,
            update=update,
            create_spatial_index=create_spatial_index,
            explodecollections=explodecollections,
            force_output_geometrytype=force_output_geometrytype,
            priority_class=priority_class,    
            sqlite_journal_mode=sqlite_journal_mode,
            verbose=verbose)
    elif gdal_to_use == 'gdal_bin':
        logger.debug("Use ogr2ogr.exe instead of python version")
        return vector_translate_exe( 
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            input_layers=input_layers,
            output_layer=output_layer,
            spatial_filter=spatial_filter,
            clip_bounds=clip_bounds,
            sql_stmt=sql_stmt,
            sql_dialect=sql_dialect_to_use,
            transaction_size=transaction_size,
            append=append,
            update=update,
            create_spatial_index=create_spatial_index,
            explodecollections=explodecollections,
            force_output_geometrytype=force_output_geometrytype,
            priority_class=priority_class,    
            sqlite_journal_mode=sqlite_journal_mode,
            verbose=verbose)
    else:
        raise Exception(f"Unsupported gdal_to_use: {gdal_to_use}")

def vector_translate_exe(
        input_path: Path, 
        output_path: Path,
        translate_description: str = None,
        input_layers: Union[Optional[List[str]], str] = None,
        output_layer: str = None,
        spatial_filter: Tuple[float, float, float, float] = None,
        clip_bounds: Tuple[float, float, float, float] = None, 
        sql_stmt: str = None,
        sql_dialect: str = None,
        transaction_size: int = 10000,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = None,
        explodecollections: bool = False,
        force_output_geometrytype: GeometryType = None,
        priority_class: str = 'VERY_LOW',
        sqlite_journal_mode: str = None,
        verbose: bool = False) -> bool:

    ##### Init #####
    if output_layer is None:
        output_layer = output_path.stem
    if os.name == 'nt':
        ogr2ogr_exe = 'ogr2ogr.exe'
    else:
        ogr2ogr_exe = 'ogr2ogr'
    
    # If GDAL_BIN is set, use ogr2ogr located there
    gdal_bin_dir = os.getenv('GDAL_BIN')
    if gdal_bin_dir is not None:
        gdal_bin_dir = Path(gdal_bin_dir)
        ogr2ogr_path = gdal_bin_dir / ogr2ogr_exe
    else:
        ogr2ogr_path = ogr2ogr_exe

    # Add all parameters to args list
    args = [str(ogr2ogr_path)]
    #if verbose:
    #    args.append('-progress')

    # Sql'ing, Filtering, clipping  
    if spatial_filter is not None:
        args.extend(['-spat', str(spatial_filter[0]), str(spatial_filter[1]), 
                    str(spatial_filter[2]), str(spatial_filter[3])])
    if clip_bounds is not None:
        args.extend(['-clipsrc', str(clip_bounds[0]), str(clip_bounds[1]), 
                    str(clip_bounds[2]), str(clip_bounds[3])])
    if sql_stmt is not None:
        args.extend(['-sql', sql_stmt])
    if sql_dialect is not None:
        args.extend(['-dialect', sql_dialect])

    # Output file options
    if output_path.exists() is True:
        if append is True:
            args.append('-append')
        if update is True:
            args.append('-update')

    # Files and input layer(s)
    args.append(str(output_path))
    args.append(str(input_path))
    if input_layers is not None:
        if isinstance(input_layers, str):
            args.append(input_layers)
        else:
            for input_layer in input_layers:
                args.append(input_layer)
            
    # Output layer options
    if explodecollections is True:
        args.append('-explodecollections')
    if output_layer is not None:
        args.extend(['-nln', output_layer])
    if force_output_geometrytype is not None:
        args.extend(['-nlt', force_output_geometrytype.name])
    args.extend(['-nlt', 'PROMOTE_TO_MULTI'])
    if create_spatial_index is not None:
        if create_spatial_index is True:
            args.extend(['-lco', 'SPATIAL_INDEX=YES'])
        else:
            args.extend(['-lco', 'SPATIAL_INDEX=NO'])
    if transaction_size is not None:
        args.extend(['-gt', str(transaction_size)])
    
    # Sqlite specific options
    '''
    # Try if the busy_timeout isn't giving problems rather than solving them...
    if sqlite_journal_mode is not None:
        args.extend(['--config', 'OGR_SQLITE_PRAGMA', f"journal_mode={sqlite_journal_mode},busy_timeout=5000"])  
    else:
        args.extend(['--config', 'OGR_SQLITE_PRAGMA', 'busy_timeout=5000'])  
    '''
    if sqlite_journal_mode is not None:
        args.extend(['--config', 'OGR_SQLITE_PRAGMA', f"journal_mode={sqlite_journal_mode}"])  

    #if append is False:
    #    args.extend(['--config', 'OGR_SQLITE_PRAGMA', '"journal_mode=WAL"'])
    #    args.extend(['-dsco', 'ADD_GPKG_OGR_CONTENTS=NO'])
    #else:
    #    args.extend(['--config', 'OGR_SQLITE_PRAGMA', 'busy_timeout=-1'])  
    #args.extend(['--config', 'OGR_SQLITE_SYNCHRONOUS', 'OFF'])  
    args.extend(['--config', 'OGR_SQLITE_CACHE', '128'])

    ##### Now start ogr2ogr #####
    # Save whether the output file exists already prior to the operation
    output_path_exists_already = output_path.exists()

    # Make sure the output dir exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Set priority of the process on Windows, so computer stays responsive
    if os.name == 'nt':  
        if priority_class is None or priority_class == 'NORMAL':
            creationflags = 0x00000020 # = NORMAL_PRIORITY_CLASS
        elif priority_class == 'VERY_LOW':
            creationflags = 0x00000040 # =IDLE_PRIORITY_CLASS
        elif priority_class == 'LOW':
            creationflags = 0x00004000 # =BELOW_NORMAL_PRIORITY_CLASS
        else: 
            raise Exception("Unsupported priority class: {priority_class}, use one of: 'NORMAL', 'LOW' or 'VERY_LOW'")
    else:
        # On non-windows os'es, creationflags cannot be used
        creationflags = 0

    # Geopackage/sqlite files are very sensitive for being locked, so retry till 
    # file is not locked anymore... 
    # TODO: ideally, the child processes would die when the parent is killed!
    returncode = None
    err = None
    for retry_count in range(10):
        if translate_description is not None:
            logger.debug(f"Start '{translate_description}' with retry_count: {retry_count}")

        # Start process!
        process = subprocess.Popen(args, 
                creationflags=creationflags,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, encoding='utf-8')
        output, err = process.communicate()
        returncode = process.returncode

        # If an error occured
        if returncode > 0:
            if str(err).startswith("ERROR 1: database is locked"):
                logger.warn(f"'ERROR 1: database is locked' occured during {translate_description}, retry nb: {retry_count}")
                time.sleep(1)
                continue
            else:
                # If output_path didn't exist yet before, clean it up... if it exists
                if not output_path_exists_already:
                    if output_path.exists():
                        geofile.remove(output_path)
                raise Exception(f"Error executing {pprint.pformat(args)}\n\t-> Return code: {returncode}\n\t-> Error: {err}\n\t->Output: {output}")
        elif(err is not None and err != ""
             and not str(err).startswith(r"Warning 1: Layer creation options ignored since an existing layer is")):
            # No error, but data (warnings) in stderr
            # Remark: the warning about layer creating options is common and not interesting: ignore!
            logger.warn(f"Finished executing {pprint.pformat(args)}")
            logger.warn(f"\t-> Returncode ok, but stderr contains: {err}")

        # Check if the output file contains data
        if output_path.exists():
            fileinfo = _getfileinfo(output_path, readonly=False)
            if len(fileinfo['layers']) == 0:
                output_path.unlink()
                if verbose is True:
                    logger.warn(f"Finished, but empty result for '{translate_description}'")
            elif translate_description is not None:
                if verbose is True:
                    logger.info(f"Finished '{translate_description}'")
                else:
                    logger.debug(f"Finished '{translate_description}'")
        else:
            if verbose is True:
                logger.warn(f"Finished, but empty result for '{translate_description}'")
                
        return True

    # If we get here, the retries didn't suffice to get it executed properly
    raise Exception(f"Error executing {pprint.pformat(args)}\n\t-> Return code: {returncode}\n\t-> Error: {err}")

def vector_translate_py(
        input_path: Path, 
        output_path: Path,
        translate_description: str = None,
        input_layers: Union[Optional[List[str]], str] = None,
        output_layer: str = None,
        spatial_filter: Tuple[float, float, float, float] = None,
        clip_bounds: Tuple[float, float, float, float] = None, 
        sql_stmt: str = None,
        sql_dialect: str = None,
        transaction_size: int = 65536,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = None,
        explodecollections: bool = False,
        force_output_geometrytype: GeometryType = None,
        priority_class: str = 'VERY_LOW',
        sqlite_journal_mode: str = None,
        verbose: bool = False) -> bool:

    # Remark: when executing a select statement, I keep getting error that 
    # there are two columns named "geom" as he doesnt see the "geom" column  
    # in the select as a geometry column. Probably a version issue. Maybe 
    # try again later.
    args = []

    # Apparently, for shapefiles, having input_layers not None gives issues...
    if input_path.suffix == '.shp':
        input_layers = None

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
    if output_path.exists() is True:
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
        args.extend(['-nlt', force_output_geometrytype.name])
    args.extend(['-nlt', 'PROMOTE_TO_MULTI'])
    if transaction_size is not None:
        args.extend(['-gt', str(transaction_size)])

    # Output layer creation options
    layerCreationOptions = []
    # TODO: should check if the layer exists instead of the file
    if not output_path.exists():
        if create_spatial_index is not None:
            if create_spatial_index is True:
                layerCreationOptions.extend(['SPATIAL_INDEX=YES'])
            else:
                layerCreationOptions.extend(['SPATIAL_INDEX=NO'])
    # Get output format from the filename
    # TODO: better te remove if not needed
    output_format = geofile.get_driver(output_path)

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
    gdal.SetConfigOption('OGR_SQLITE_CACHE', '128')

    options = gdal.VectorTranslateOptions(
            options=args, 
            format=output_format, 
            accessMode=None, 
            srcSRS=None, 
            dstSRS=None, 
            reproject=False, 
            SQLStatement=sql_stmt,
            SQLDialect=sql_dialect,
            where=None, #"geom IS NOT NULL", 
            selectFields=None, 
            addFields=False, 
            forceNullable=False, 
            spatFilter=spatial_filter, 
            spatSRS=None,
            datasetCreationOptions=datasetCreationOptions, 
            layerCreationOptions=layerCreationOptions, 
            layers=input_layers,
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
        logger.debug(f"Execute {sql_stmt} on {input_path}")
        input_ds = gdal.OpenEx(str(input_path))

        # TODO: memory output support might be interesting to support
        result_ds = gdal.VectorTranslate(
                destNameOrDestDS=str(output_path),
                srcDS=input_ds,
                #SQLStatement=sql_stmt,
                #SQLDialect=sql_dialect,
                #layerName=output_layer
                options=options
                )
        if result_ds is None:
            raise Exception("BOEM")
        else:
            if result_ds.GetLayerCount() == 0:
                del result_ds
                geofile.remove(output_path)
    except Exception as ex:
        message = f"Error executing {sql_stmt}"
        logger.exception(message)
        raise Exception(message) from ex
    finally:
        if input_ds is not None:
            del input_ds
        
    return True

def _getfileinfo(
        path: Path,
        readonly: bool = True,
        verbose: bool = False) -> dict:
            
    # Get info
    info_str = vector_info(
            path=path, 
            readonly=readonly,
            verbose=verbose)

    # Prepare result
    result_dict = {}
    result_dict['info_str'] = info_str
    result_dict['layers'] = []
    info_strio = StringIO(str(info_str))
    for line in info_strio.readlines():
        line = line.strip()
        if re.match(r"\A\d: ", line):
            # It is a layer, now extract only the layer name
            logger.debug(f"This is a layer: {line}")
            layername_with_geomtype = re.sub(r"\A\d: ", "", line)
            layername = re.sub(r" [(][a-zA-Z ]+[)]\Z", "", layername_with_geomtype)
            result_dict['layers'].append(layername)

    return result_dict

'''
def vector_info(
        path: Path, 
        task_description = None,
        layer: str = None,
        readonly: bool = False,
        report_summary: bool = False,
        sql_stmt: str = None,
        sql_dialect: str = None, 
        skip_health_check: bool = False,
        force_py: bool = False,      
        verbose: bool = False):
    """Run a command"""

    ##### Init #####
    if not path.exists():
        raise Exception(f"File does not exist: {path}")

    sql_dialect_to_use = sql_dialect
    if force_py:
        gdal_to_use = 'gdal_default'
    elif(sql_stmt is not None 
       and sql_dialect is not None
       and sql_dialect.upper() == 'SQLITE' 
       and path.suffix.lower() == '.gpkg'):
        gdal_to_use, sql_dialect_to_use = get_gdal_to_use(sql_stmt)
    else:
        gdal_to_use = 'gdal_default'

    if gdal_to_use == 'gdal_default':
        vector_info_py(
                path=path, 
                task_description=task_description,
                layer=layer,
                readonly=readonly,
                report_summary=report_summary,
                sql_stmt=sql_stmt,
                sql_dialect=sql_dialect_to_use, 
                skip_health_check=skip_health_check,      
                verbose=verbose)
    elif gdal_to_use == 'gdal_bin':
        vector_info_exe(
                path=path, 
                task_description=task_description,
                layer=layer,
                readonly=readonly,
                report_summary=report_summary,
                sql_stmt=sql_stmt,
                sql_dialect=sql_dialect, 
                skip_health_check=skip_health_check,      
                verbose=verbose)
    else:
        raise Exception(f"Unsupported gdal_to_use: {gdal_to_use}")

def vector_info_py(
        path: Path, 
        task_description = None,
        layer: str = None,
        readonly: bool = False,
        report_summary: bool = False,
        sql_stmt: str = None,
        sql_dialect: str = None, 
        skip_health_check: bool = False,      
        verbose: bool = False):
    """"Run a command"""

    ##### Init #####
    if not path.exists():
        raise Exception(f"File does not exist: {path}")

    if(skip_health_check is False
       and sql_stmt is not None 
       and sql_dialect is not None
       and sql_dialect.upper() == 'SQLITE' 
       and path.suffix.lower() == '.gpkg'):
        _, sql_dialect = get_gdal_to_use(sql_stmt)

        # Warn that updates with 'INDIRECT_SQLITE' give terrible performance!
        if sql_dialect == 'INDIRECT_SQLITE':
            logger.warn(f"sql_stmt needs 'INDIRECT_SQLITE', but this is gonna be slow!\n\t{sql_stmt}")  

    # Add all parameters to args list
    args = []
    #args.extend(['--config', 'OGR_SQLITE_PRAGMA', 'journal_mode=WAL'])  
    if readonly is True:
        args.append('-ro')
    if report_summary is True:
        args.append('-so')
    if sql_stmt is not None:
        args.extend(['-sql', sql_stmt])
    if sql_dialect is not None:
        args.extend(['-dialect', sql_dialect])

    # File and optionally the layer
    args.append(str(path))
    if layer is not None:
        # ogrinfo doesn't like + need quoted layer names, so remove single and double quotes
        layer_stripped = layer.strip("'\"")
        args.append(layer_stripped)

    info_options = gdal.InfoOptions(
            options=args, 
            format='text', 
            deserialize=True, 
            computeMinMax=False, 
            reportHistograms=False, 
            reportProj4=False, 
            stats=False, 
            approxStats=False, 
            computeChecksum=False, 
            showGCPs=True, 
            showMetadata=True, 
            showRAT=True, 
            showColorTable=True, 
            listMDD=False, 
            showFileList=True, 
            allMetadata=False, 
            extraMDDomains=None, 
            wktFormat=None)

    result = gdal.Info(
            ds=str(path),
            options=info_options)
'''

def vector_info(
        path: Path, 
        task_description = None,
        layer: str = None,
        readonly: bool = False,
        report_summary: bool = False,
        sql_stmt: str = None,
        sql_dialect: str = None, 
        skip_health_check: bool = False,      
        verbose: bool = False):
    """"Run a command"""

    ##### Init #####
    if not path.exists():
        raise Exception(f"File does not exist: {path}")
    if os.name == 'nt':
        ogrinfo_exe = 'ogrinfo.exe'
    else:
        ogrinfo_exe = 'ogrinfo'
    
    # If GDAL_BIN is set, use ogrinfo located there
    gdal_bin_dir = os.getenv('GDAL_BIN')
    if gdal_bin_dir is not None:
        gdal_bin_dir = Path(gdal_bin_dir)
        ogrinfo_path = gdal_bin_dir / ogrinfo_exe
    else:
        ogrinfo_path = ogrinfo_exe

    # If running a sqlite statement on a gpkg file, check first if spatialite 
    # works properly an if INDIRECT_SQLITE should be used.
    if(skip_health_check is False
       and sql_stmt is not None 
       and sql_dialect is not None
       and sql_dialect.upper() == 'SQLITE' 
       and path.suffix.lower() == '.gpkg'):
        _, sql_dialect = get_gdal_to_use(sql_stmt)

        # Warn that updates with 'INDIRECT_SQLITE' give terrible performance!
        if sql_dialect == 'INDIRECT_SQLITE':
            logger.warn(f"sql_stmt needs 'INDIRECT_SQLITE', but this is gonna be slow!\n\t{sql_stmt}")  

    # Add all parameters to args list
    args = [str(ogrinfo_path)]
    #args.extend(['--config', 'OGR_SQLITE_PRAGMA', 'journal_mode=WAL'])  
    if readonly is True:
        args.append('-ro')
    if report_summary is True:
        args.append('-so')
    if sql_stmt is not None:
        args.extend(['-sql', sql_stmt])
    if sql_dialect is not None:
        args.extend(['-dialect', sql_dialect])

    # File and optionally the layer
    args.append(str(path))
    if layer is not None:
        # ogrinfo doesn't like + need quoted layer names, so remove single and double quotes
        layer_stripped = layer.strip("'\"")
        args.append(layer_stripped)

    # TODO: ideally, the child processes would die when the parent is killed!
    # Start on low priority, so computer stays responsive
    if os.name == 'nt':
        #BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        creationflags = 0x00000040
    else:
        # On linux, needs to be 0, otherwise error!
        creationflags = 0

    ##### Run ogrinfo #####
    # Geopackage/sqlite files are very sensitive for being locked, so retry till 
    # file is not locked anymore...
    sleep_time = 1
    returncode = None 
    for retry_count in range(10):
        process = subprocess.Popen(args, 
                creationflags=creationflags,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, encoding='utf-8')
        output, err = process.communicate()
        returncode = process.returncode

        # If an error occured
        if returncode > 0:
            if str(err).startswith("ERROR 1: database is locked"):
                logger.warn(f"'ERROR 1: database is locked' occured during {task_description}, retry nb: {retry_count}")
                time.sleep(sleep_time)
                sleep_time += 1
                continue
            else:
                raise Exception(f"Error executing {pprint.pformat(args)}\n\t-> Return code: {returncode}\n\t-> Error: {err}\n\t->Output: {output}")  
        elif err is not None and err != "":
            # ogrinfo apparently sometimes give a wrong returncode, so if data 
            # in stderr, treat as error as well
            raise Exception(f"Error executing {pprint.pformat(args)}\n\t->Return code: {returncode}\n\t->Error: {err}\n\t->Output: {output}")
        elif verbose is True:
            logger.info(f"Ready executing {pprint.pformat(args)}")

        return output

    # If we get here, the retries didn't suffice to get it executed properly
    raise Exception(f"Error executing {pprint.pformat(args)}\n\t-> Return code: {returncode}")

def get_gdal_to_use(sqlite_stmt: str) -> Tuple[str, str]:
    """
    Check which gdal installation to use for this query: 
      * If query is supported using standard python execution, returns 'gdal_default'
      * If query is only supported using gedal installed in GDAL_BIN, returns 'gdal_bin'
      * If query is not supported at all, raises exception

    Raises:
        SQLNotSupportedException: [description]
    """
    ### First check if the python version can be used ###
    # Get spatialite version info for the default gdal 
    global global_spatiallite_versioninfo_default
    if global_spatiallite_versioninfo_default is None:
        global_spatiallite_versioninfo_default = get_gdal_install_info('gdal_default')

    # If version 5 and rttopo installed... the default python way can be used!
    if(global_spatiallite_versioninfo_default['spatialite_version()'] >= '5.0' 
       and global_spatiallite_versioninfo_default['rttopo_version()'] is not None):
        return ('gdal_default', 'SQLITE')
        
    # Check if there is an unsupported function in the sqlite statement
    sqlite_stmt_lower = sqlite_stmt.lower()
    unsupported_function_found = False 
    for unsupported_function in global_spatiallite_versioninfo_default['unsupported_functions']:
        if(f"{unsupported_function}(" in sqlite_stmt_lower
           or f"{unsupported_function} (" in sqlite_stmt_lower):
            unsupported_function_found = True
            break

    # If no unsupported function found, default gdal can be used
    if unsupported_function_found is False:
        return ('gdal_default', 'SQLITE')
    else:
        # If no spatial index table is used in sql_stmt, we can force indirect sql 
        # TODO: could be checked in a more robust way...
        if not 'join "rtree_' in sqlite_stmt_lower:
            return ('gdal_default', 'INDIRECT_SQLITE')

    ### Default gdal not OK, so check if there is a GDAL_BIN version that is ok ###
    # First check if there is an alternative gdal installation 
    gdal_bin_dir = os.getenv('GDAL_BIN')
    if gdal_bin_dir is None:
        raise MissingRuntimeDependencyError('sqlite_stmt not supported by default gdal, and no alternative gdal installation specified using GDAL_BIN')

    # Get spatiallite version info for the alternative gdal, if it isn't cached yet for this gdal_bin_dir
    global global_spatiallite_versioninfo_alternative
    if global_spatiallite_versioninfo_alternative is None:
        global_spatiallite_versioninfo_alternative = get_gdal_install_info('gdal_bin')
        global_spatiallite_versioninfo_alternative['gdal_bin_dir'] = gdal_bin_dir
    elif global_spatiallite_versioninfo_alternative['gdal_bin_dir'] != gdal_bin_dir:
        global_spatiallite_versioninfo_alternative = get_gdal_install_info('gdal_bin')
        global_spatiallite_versioninfo_alternative['gdal_bin_dir'] = gdal_bin_dir

    # Check if there is an unsupported function in the sqlite statement
    sqlite_stmt_lower = sqlite_stmt.lower()
    unsupported_function_found = False 
    for unsupported_function in global_spatiallite_versioninfo_alternative['unsupported_functions']:
        if(f"{unsupported_function}(" in sqlite_stmt_lower
           or f"{unsupported_function} (" in sqlite_stmt_lower):
            unsupported_function_found = True
            break

    # If no unsupported function found, alternative gdal can be used
    if unsupported_function_found is False:
        return ('gdal_bin', 'SQLITE')
    else:
        raise MissingRuntimeDependencyError(f"Alternative gdal in gdal_bin_dir {gdal_bin_dir} doesn't support sqlite_stmt either: {sqlite_stmt}")

def get_gdal_install_info(gdal_installation: str) -> dict:

    install_info = {}
    
    # First check the spatialite version
    test_path = Path(__file__).resolve().parent / "test.gpkg"
    sqlite_stmt = f'select spatialite_version()'    
    install_info_gdf = _execute_sql(test_path, sqlite_stmt, gdal_installation, sql_dialect='SQLITE')
        
    # Now get extra information, depending on the spatialite version
    if install_info_gdf['spatialite_version()'].values[0] >= '5.0.0':
        sqlite_stmt = f'select spatialite_version(), HasGeos(), HasGeosAdvanced(), HasGeosTrunk(), geos_version(), rttopo_version()'
    elif install_info_gdf['spatialite_version()'].values[0] >= '4.3.0':
        sqlite_stmt = f'select spatialite_version(), HasGeos(), HasGeosAdvanced(), HasGeosTrunk(), geos_version(), lwgeom_version()'
    else:
        raise Exception(f"Unsupported spatialite version: {install_info_gdf['spatialite_version()'][0]}")
    install_info_gdf = _execute_sql(test_path, sqlite_stmt, gdal_installation, sql_dialect='SQLITE')
    
    # Copy results to result dict
    for column in install_info_gdf.columns:
        install_info[column] = install_info_gdf[column][0]

    # Check if there are unsupported functions
    if install_info['spatialite_version()'] >= '5.0.0':
        if install_info['rttopo_version()'] is None:
            install_info['unsupported_functions'] = [
                    'makevalid', 'st_makevalid', 'isvalid', 'st_isvalid', 
                    'isvalidreason', 'st_isvalidreason', 'isvaliddetail', 'st_isvaliddetail']
        else:
            install_info['unsupported_functions'] = []
    elif install_info['spatialite_version()'] >= '4.3.0':
        if install_info['lwgeom_version()'] is None:
            install_info['unsupported_functions'] = [
                    'buffer', 'st_buffer', 
                    'convexhull', 'st_convexhull', 'simplify', 'st_simplify', 
                    'simplifypreservetopology', 'st_simplifypreservetopology', 
                    'makevalid', 'st_makevalid', 'isvalid', 'st_isvalid', 
                    'isvalidreason', 'st_isvalidreason', 'isvaliddetail', 'st_isvaliddetail',
                    'st_area', 'area', 'st_npoints', 'npoints',
                    'st_multi', 'collectionextract', 'st_intersection', 'st_union', 'st_collect']
        else:
            install_info['unsupported_functions'] = []

    return install_info

def _execute_sql(
        path: Path,
        sqlite_stmt: str,
        gdal_installation: str,
        sql_dialect: str) -> gpd.GeoDataFrame:
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / 'ogr_util_execute_sql_tmp_file.gpkg'
        if gdal_installation == 'gdal_default':
            vector_translate_py(
                    input_path=path,
                    output_path=tmp_path,
                    sql_stmt=sqlite_stmt,
                    sql_dialect=sql_dialect)
        elif gdal_installation == 'gdal_bin':
            vector_translate_exe(
                    input_path=path,
                    output_path=tmp_path,
                    sql_stmt=sqlite_stmt,
                    sql_dialect=sql_dialect)
        else:
            raise Exception(f"Unsupported gdal_installation: {gdal_installation}")
        
        # Return result
        install_info_gdf = geofile.read_file(tmp_path)
        return install_info_gdf

if __name__ == '__main__':
    None
