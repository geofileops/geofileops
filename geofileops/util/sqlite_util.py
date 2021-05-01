# -*- coding: utf-8 -*-
"""
Module containing utilities regarding the usage of ogr functionalities.
"""

import datetime
import enum
import logging
import os
from pathlib import Path
import sqlite3

from geofileops.util.geometry_util import GeometryType
from geofileops import geofile
from .general_util import MissingRuntimeDependencyError

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def check_runtimedependencies():
    test_path = Path(__file__).resolve().parent / "test.gpkg"
    conn = sqlite3.connect(test_path)
    load_spatialite(conn)

class SqliteProfile(enum.Enum):
    DEFAULT=0
    SPEED=1

def create_table_as_sql(
        input1_path: Path,
        input1_layer: str,
        input2_path: Path, 
        output_path: Path,
        sql_stmt: str,
        output_layer: str,
        output_geometrytype: GeometryType,
        append: bool = False,
        update: bool = False,
        create_spatial_index: bool = True,
        profile: SqliteProfile = SqliteProfile.DEFAULT):

    # Check input parameters
    if append is True or update is True:
        raise Exception('Not implemented')

    # Use crs epsg from input1_layer, if it has one
    input1_layerinfo = geofile.get_layerinfo(input1_path, input1_layer)
    crs_epsg = -1
    if input1_layerinfo.crs is not None and input1_layerinfo.crs.to_epsg() is not None: 
        crs_epsg = input1_layerinfo.crs.to_epsg()
        
    sql = None
    try:
        def to_string_for_sql(input) -> str:
            if input is None:
                return 'NULL'
            else:
                return str(input)

        # Connect to output database file (by convention) + init
        output_databasename = 'main'
        with sqlite3.connect(output_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            load_spatialite(conn)
            
            # Set number of cache pages (1 page = 4096 bytes)
            conn.execute('PRAGMA cache_size = 10000;')
            conn.execute('PRAGMA temp_store = MEMORY;')
            
            # Use the sqlite profile specified
            if profile is SqliteProfile.SPEED:
                conn.execute(f"PRAGMA journal_mode = OFF;")
            
                # These pragma's increase speed
                conn.execute('PRAGMA locking_mode = EXCLUSIVE;')
                conn.execute('PRAGMA synchronous = OFF;')
                
                # Use memory mapped IO = much faster (max 30GB)
                conn.execute('PRAGMA mmap_size = 30000000000;')

            # Init as gpkg
            sql = 'SELECT gpkgCreateBaseTables();'
            conn.execute(sql)
            sql = 'SELECT EnableGpkgMode();'
            conn.execute(sql)
            if crs_epsg is not None and crs_epsg not in [0, -1]:
                sql = f"SELECT gpkgInsertEpsgSRID({crs_epsg})"
                conn.execute(sql)

            # If input1 isn't the same database as output, attach to it
            if input1_path == output_path:
                input1_databasename = output_databasename
            else:
                input1_databasename = 'input1'
                sql = f"ATTACH DATABASE ? AS {input1_databasename}"
                dbSpec = (str(input1_path),)
                conn.execute(sql, dbSpec)
            
            # If input2 isn't the same database as output or as input1, attach to it
            if input2_path == output_path:
                input2_databasename = output_databasename
            elif input2_path == input1_path:
                input2_databasename = input1_databasename
            else:
                input2_databasename = 'input2'
                sql = f"ATTACH DATABASE ? AS {input2_databasename}"
                dbSpec = (str(input2_path),)
                conn.execute(sql, dbSpec)
            
            # Prepare sql statement
            sql_stmt = sql_stmt.format(
                    input1_databasename=input1_databasename,
                    input2_databasename=input2_databasename)

            ### Determine columns/datatypes to create the table ###
            # Create temp table to get the column names + general data types
            sql = f'CREATE TEMPORARY TABLE tmp AS \n{sql_stmt}\nLIMIT 0;'
            conn.execute(sql)
            sql = 'PRAGMA TABLE_INFO(tmp)'
            cur = conn.execute(sql)
            tmpcolumns = cur.fetchall()
                        
            # Fetch one row to try to get more detailed data types
            sql = sql_stmt
            tmpdata = conn.execute(sql).fetchone()
            
            # Loop over all columns to determine the data type
            column_types = {}
            for column_index, column in enumerate(tmpcolumns):
                columnname = column[1]
                columntype = column[2]

                if columnname == 'geom':
                    # PRAGMA TABLE_INFO gives None as column type for a 
                    # geometry column. So if output_geometrytype not specified, 
                    # Use ST_GeometryType to get the type
                    # based on the data + apply to_multitype to be sure
                    if output_geometrytype is None:
                        sql = f'SELECT ST_GeometryType({columnname}) FROM tmp;'
                        column_geometrytypename = conn.execute(sql).fetchall()[0][0]
                        output_geometrytype = GeometryType[column_geometrytypename].to_multitype
                    column_types[columnname] = output_geometrytype.name
                else:
                    # If PRAGMA TABLE_INFO doesn't specify the datatype, 
                    # determine based on data
                    if columntype is None or columntype == '':
                        sql = f'SELECT typeof({columnname}) FROM tmp;'
                        column_types[columnname] = conn.execute(sql).fetchall()[0][0]
                    elif columntype == 'NUM':
                        # PRAGMA TABLE_INFO sometimes returns 'NUM', but 
                        # apparently this cannot be used in "CREATE TABLE" 
                        if isinstance(tmpdata[column_index], datetime.date): 
                            column_types[columnname] = 'DATE'
                        elif isinstance(tmpdata[column_index], datetime.datetime): 
                            column_types[columnname] = 'DATETIME'
                        else:
                            column_types[columnname] = 'DECIMAL'
                    else:
                        column_types[columnname] = columntype

            ### Now we can create the table ###
            # Create output table using the gpkgAddGeometryColumn() function
            # Problem: the spatialite function gpkgAddGeometryColumn() doesn't support 
            # layer names with special characters (eg. '-')... 
            # Solution: mimic the behaviour of gpkgAddGeometryColumn manually.
            # Create table without geom column
            ''' 
            columns_for_create = [f"{columnname} {column_types[columnname]}\n" for columnname in column_types if columnname != 'geom']
            sql = f'CREATE TABLE {output_databasename}."{output_layer}" ({", ".join(columns_for_create)})'
            conn.execute(sql)
            # Add geom column with gpkgAddGeometryColumn()
            # Remark: output_geometrytype.name should be detemined from data if needed, see above...
            sql = f"SELECT gpkgAddGeometryColumn('{output_layer}', 'geom', '{output_geometrytype.name}', 0, 0, {to_string_for_sql(crs_epsg)});"
            conn.execute(sql)
            '''
            columns_for_create = [f"{columnname} {column_types[columnname]}\n" for columnname in column_types]
            sql = f'CREATE TABLE {output_databasename}."{output_layer}" ({", ".join(columns_for_create)})'
            conn.execute(sql)
            # Add metadata (~ mimic behaviour of gpkgAddGeometryColumn())
            sql = f"""
                    INSERT INTO {output_databasename}.gpkg_contents (
                            table_name, data_type, identifier, description, last_change, 
                            min_x, min_y, max_x, max_y, srs_id)
                        VALUES ('{output_layer}', 'features', NULL, '', DATETIME(),
                            NULL, NULL, NULL, NULL, {to_string_for_sql(crs_epsg)});"""
            conn.execute(sql)
            sql = f"""
                    INSERT INTO {output_databasename}.gpkg_geometry_columns (
                            table_name, column_name, geometry_type_name, srs_id, z, m)
                        VALUES ('{output_layer}', 'geom', '{output_geometrytype.name}', {to_string_for_sql(crs_epsg)}, 0, 0);"""
            conn.execute(sql)
            
            # Now add geom triggers
            sql = f"SELECT gpkgAddGeometryTriggers('{output_layer}', 'geom');"
            cur.execute(sql)
    
            # Insert data using the sql statement specified
            columns_for_insert = [f'"{column[1]}"' for column in tmpcolumns]
            sql = f'INSERT INTO {output_databasename}."{output_layer}" ({", ".join(columns_for_insert)})\n{sql_stmt}' 
            conn.execute(sql)           
                    
            # Create spatial index if needed
            if create_spatial_index is True:
                sql = f"SELECT UpdateLayerStatistics('{output_layer}', 'geom');"
                cur.execute(sql)
                sql = f"SELECT gpkgAddSpatialIndex('{output_layer}', 'geom');"
                cur.execute(sql)
            conn.commit()
            
    except Exception as ex:
        raise Exception(f"Error executing {sql}") from ex

def load_spatialite(conn):
    """
    Load mod_spatialite for an existing sqlite connection.

    Args:
        conn ([type]): Sqlite connection
    """
    conn.enable_load_extension(True)
    try:
        conn.load_extension('mod_spatialite')
    except Exception as ex:
        mod_spatialite_dir = os.environ.get('MOD_SPATIALITE_DIR')
        if mod_spatialite_dir is not None:
            if f"{mod_spatialite_dir};" not in os.environ['PATH']:
                os.environ['PATH'] = f"{mod_spatialite_dir};{os.environ['PATH']}"
            try:
                conn.load_extension('mod_spatialite')
            except Exception as ex:
                raise MissingRuntimeDependencyError(f"Error trying to load mod_spatialite with MOD_SPATIALITE_DIR: {mod_spatialite_dir}") from ex
        else:
            raise MissingRuntimeDependencyError("Error trying to load mod_spatialite. You can specify the location of mod_spatialite using the MOD_SPATIALITE_DIR environment variable") from ex
    
if __name__ == '__main__':
    None
