# -*- coding: utf-8 -*-
"""
Module containing utilities regarding the usage of ogr functionalities.
"""

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

    if append is True or update is True:
        raise Exception('Not implemented') 
    # Use crs from input1_layer:
    input1_layerinfo = geofile.get_layerinfo(input1_path, input1_layer)
    crs_epsg = input1_layerinfo.crs.to_epsg()
    if crs_epsg is None:
        crs_epsg = -1        
        if input1_layerinfo.crs is not None:
            if input1_layerinfo.crs.name == 'Belge 1972 / Belgian Lambert 72':
                crs_epsg = 31370
            else:
                logger.warning(f"no epsg code found for crs, so -1 used: {input1_layerinfo.crs}")

    sql = None
    try:
        # Connect to output database file (by convention) + init
        output_databasename = 'main'
        with sqlite3.connect(output_path) as conn:
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
            if crs_epsg != -1:
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

            # Create temp table to get the date types
            sql = f'CREATE TEMPORARY TABLE tmp AS \n{sql_stmt}\nLIMIT 0;'
            conn.execute(sql)

            cur = conn.execute('PRAGMA TABLE_INFO(tmp)')
            columns = cur.fetchall()
            
            # Create output table using the gpkgAddGeometryColumn() function
            # Problem: the spatialite function gpkgAddGeometryColumn() doesn't support 
            # layer names with special characters (eg. '-')... 
            # Solution: mimic the behaviour of gpkgAddGeometryColumn manually.
            # Create table without geom column
            '''
            columns_for_create = [f'"{column[1]}" {column[2]}\n' for column in columns if column[1] != 'geom']
            sql = f'CREATE TABLE {output_databasename}."{output_layer}" ({", ".join(columns_for_create)})'
            conn.execute(sql)
            # Add geom column with gpkgAddGeometryColumn()
            sql = f"SELECT gpkgAddGeometryColumn('{output_layer}', 'geom', '{output_geometrytype.name}', 0, 0, {crs_epsg});"
            conn.execute(sql)
            '''
            # Create table
            columns_for_create = [
                    f'"{column[1]}" {output_geometrytype.name}\n' if column[1] == 'geom' else f'"{column[1]}" {column[2]}\n' 
                    for column in columns]
            sql = f'CREATE TABLE {output_databasename}."{output_layer}" ({", ".join(columns_for_create)})'
            conn.execute(sql)
            # Add metadata (~ mimic behaviour of gpkgAddGeometryColumn())
            sql = f"""
                    INSERT INTO {output_databasename}.gpkg_contents (
                            table_name, data_type, identifier, description, last_change, 
                            min_x, min_y, max_x, max_y, srs_id)
                        VALUES ('{output_layer}', 'features', NULL, '', DATETIME(),
                            NULL, NULL, NULL, NULL, {crs_epsg});"""
            conn.execute(sql)
            sql = f"""
                    INSERT INTO {output_databasename}.gpkg_geometry_columns (
                            table_name, column_name, geometry_type_name, srs_id, z, m)
                        VALUES ('{output_layer}', 'geom', '{output_geometrytype.name}', {crs_epsg}, 0, 0);"""
            conn.execute(sql)
            
            # Now add geom triggers
            sql = f"SELECT gpkgAddGeometryTriggers('{output_layer}', 'geom');"
            cur.execute(sql)
    
            # Insert data using the sql statement specified
            columns_for_insert = [f'"{column[1]}"' for column in columns]
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
