#-------------------------------------
# Import/init needed modules
#-------------------------------------
from concurrent import futures
import logging
import multiprocessing
import os
from pathlib import Path
import pprint
import subprocess
from threading import Lock
import time
from typing import Any, List, Optional, Tuple, Union
from io import StringIO
import re

from osgeo import gdal
gdal.UseExceptions() 

from geofileops import geofile

# TODO: on windows, the init of this doensn't seem to work properly... should be solved somewhere else?
if os.name == 'nt':
    os.environ["GDAL_DATA"] = r"C:\Tools\miniconda3\envs\geofileops\Library\share\gdal"
    os.environ["PROJ_LIB"] = r"C:\Tools\miniconda3\envs\geofileops\Library\share\proj"

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

# Initialize the location of the GDAL binaries
ogr2ogr_exe = 'ogr2ogr.exe'
ogrinfo_exe = 'ogrinfo.exe'
gdal_bin_dir = os.getenv('GDAL_BIN')
if gdal_bin_dir is not None:
    gdal_bin_dir = Path(gdal_bin_dir)
    ogr2ogr_exe = gdal_bin_dir / ogr2ogr_exe
    ogrinfo_exe = gdal_bin_dir / ogrinfo_exe

lock = Lock()

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
            force_output_geometrytype: str = None,
            priority_class: str = 'VERY_LOW',
            sqlite_journal_mode: str = 'WAL',
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
        nb_parallel: int = -1):

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
        for future in futures.as_completed(future_to_calc_id):
            calc_id = future_to_calc_id[future]
            try:
                _ = future.result()
            except Exception as ex:
                message = f"Async translate {calc_id} ERROR: {vector_translate_infos[calc_id].translate_description}\n{pprint.pformat(vector_translate_infos[calc_id])}"
                raise Exception(message) from ex        

'''
def run_command_async(
    command: str, 
    ignorecrash: bool = False,
    verbose: bool = False):
""""Run a command"""
# Run the command
if verbose:
    logger.info(f"Command to execute: {command}")
    print(f"Command to execute: {command}")

command_process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stdin=open(os.devnull), 
        stderr=subprocess.STDOUT, universal_newlines=True)
    
return command_process
'''

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
        force_output_geometrytype: str = None,
        priority_class: str = 'VERY_LOW',
        sqlite_journal_mode: str = 'WAL',
        force_py: bool = False,
        verbose: bool = False) -> bool:
    """
    Run a command
    """
    if os.getenv('GDAL_BIN') is None or force_py == True:
        return vector_translate_py( 
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            input_layers=input_layers,
            output_layer=output_layer,
            spatial_filter=spatial_filter,
            clip_bounds=clip_bounds,
            sql_stmt=sql_stmt,
            sql_dialect=sql_dialect,
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
        return vector_translate_exe( 
            input_path=input_path,
            output_path=output_path,
            translate_description=translate_description,
            input_layers=input_layers,
            output_layer=output_layer,
            spatial_filter=spatial_filter,
            clip_bounds=clip_bounds,
            sql_stmt=sql_stmt,
            sql_dialect=sql_dialect,
            transaction_size=transaction_size,
            append=append,
            update=update,
            create_spatial_index=create_spatial_index,
            explodecollections=explodecollections,
            force_output_geometrytype=force_output_geometrytype,
            priority_class=priority_class,    
            sqlite_journal_mode=sqlite_journal_mode,
            verbose=verbose)

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
        transaction_size: int = 65536,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = None,
        explodecollections: bool = False,
        force_output_geometrytype: str = None,
        priority_class: str = 'VERY_LOW',
        sqlite_journal_mode: str = 'WAL',
        verbose: bool = False) -> bool:

    ##### Init #####
    if output_layer is None:
        output_layer = output_path.stem
    if input_layers is not None:
        logger.warn(f"input_layers is not None, but isn't used in vector_translate_exe: {input_layers}")
        
    # Add all parameters to args list
    args = [str(ogr2ogr_exe)]
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
    if append is True:
        args.append('-append')
    if update is True:
        args.append('-update')

    # Files
    args.append(str(output_path))
    args.append(str(input_path))

    # Output layer options
    if explodecollections is True:
        args.append('-explodecollections')
    if output_layer is not None:
        args.extend(['-nln', output_layer])
    if force_output_geometrytype is not None:
        args.extend(['-nlt', force_output_geometrytype])
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
    args.extend(['--config', 'OGR_SQLITE_CACHE', '512'])

    ##### Now start ogr2ogr #####
    # Save whether the output file exists already prior to the operation
    output_path_exists_already = output_path.exists()

    # Set priority of the process, so computer stays responsive
    if priority_class is None or priority_class == 'NORMAL':
        priority_class_windows = 0x00000020 # = NORMAL_PRIORITY_CLASS
    elif priority_class == 'VERY_LOW':
        priority_class_windows = 0x00000040 # =IDLE_PRIORITY_CLASS
    elif priority_class == 'LOW':
        priority_class_windows = 0x00004000 # =BELOW_NORMAL_PRIORITY_CLASS
    else: 
        raise Exception("Unsupported priority class: {priority_class}, use one of: 'NORMAL', 'LOW' or 'VERY_LOW'")

    # Geopackage/sqlite files are very sensitive for being locked, so retry till 
    # file is not locked anymore... 
    # TODO: ideally, the child processes would die when the parent is killed!
    returncode = None
    err = None
    for retry_count in range(10):

        # Unsuccessfull test to print output constantly instead of only when the 
        # process has ended. Because progress on a geopackage doesn't work 
        # anyway the use case isn't relevant anymore either
        # TODO: remove code after first check-in
        '''
        output = ""
        err = ""
        returncode = -1
        with subprocess.Popen(args, 
                creationflags=IDLE_PRIORITY_CLASS,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, 
                encoding='utf-8', universal_newlines=True) as process:
            for line in process.stdout.readline():
                if verbose:
                    logger.info(line)
                else:
                    output += line
            err = process.stderr.read()
            returncode = process.wait()
        '''
        if translate_description is not None:
            if verbose is True:
                logger.info(f"Start '{translate_description}' with retry_count: {retry_count}")
            else:
                logger.debug(f"Start '{translate_description}' with retry_count: {retry_count}")
        process = subprocess.Popen(args, 
                creationflags=priority_class_windows,
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
                # If output_path didn't exist yet before, clean it up
                if not output_path_exists_already:
                    # TODO: for shape files maybe all files need to be cleaned up?
                    output_path.unlink()
                raise Exception(f"Error executing {pprint.pformat(args)}\n\t-> Return code: {returncode}\n\t-> Error: {err}\n\t->Output: {output}")
        elif(err is not None and err != ""
             and not str(err).startswith(r"Warning 1: Layer creation options ignored since an existing layer is")):
            # No error, but data (warnings) in stderr
            # Remark: the warning about layer creating options is common and not interesting: ignore!
            logger.warn(f"Finished executing {pprint.pformat(args)}")
            logger.warn(f"\t-> Returncode ok, but stderr contains: {err}")

        # Check if the output file contains data
        fileinfo = _getfileinfo(output_path, readonly=False)
        if len(fileinfo['layers']) == 0:
            output_path.unlink()
            logger.warn(f"Finished, but empty result for '{translate_description}'")
        elif verbose is True:
            logger.info(f"Finished '{translate_description}'")
        else:
            logger.debug(f"Finished '{translate_description}'")

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
        force_output_geometrytype: str = None,
        priority_class: str = 'VERY_LOW',
        sqlite_journal_mode: str = None,
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
    gdal.SetConfigOption('OGR_SQLITE_CACHE', '512')

    options = gdal.VectorTranslateOptions(
            options=args, 
            format=output_format, 
            accessMode=None, 
            srcSRS=None, 
            dstSRS=None, 
            reproject=False, 
            SQLStatement=sql_stmt,
            SQLDialect=sql_dialect,
            where=None, 
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
        if verbose:
            logger.info(f"Execute {sql_stmt} on {input_path}")
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
    info_str = _vector_info(
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

def _vector_info(
        path: Path, 
        task_description = None,
        layer: str = None,
        readonly: bool = False,
        report_summary: bool = False,
        sql_stmt: str = None,
        sql_dialect: str = None,        
        verbose: bool = False):
    """"Run a command"""

    ##### Init #####
    if not path.exists():
        raise Exception(f"File does not exist: {path}")

    # Add all parameters to args list
    args = [str(ogrinfo_exe)]
    args.extend(['--config', 'OGR_SQLITE_PRAGMA', 'journal_mode=WAL'])  
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
    #BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    IDLE_PRIORITY_CLASS = 0x00000040

    ##### Run ogrinfo #####
    # Geopackage/sqlite files are very sensitive for being locked, so retry till 
    # file is not locked anymore...
    sleep_time = 1
    returncode = None 
    for retry_count in range(10):
        process = subprocess.Popen(args, 
                creationflags=IDLE_PRIORITY_CLASS,
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

if __name__ == '__main__':
    None
        