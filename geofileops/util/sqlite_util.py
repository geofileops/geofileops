# -*- coding: utf-8 -*-
"""
Module containing utilities regarding the usage of ogr functionalities.
"""

import logging
from pathlib import Path
import sqlite3

from geofileops.util.geometry_util import GeometryType
from geofileops import geofile

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------

# Get a logger...
logger = logging.getLogger(__name__)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

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
        journal_mode: str = None):

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

    sql = sql_stmt
    try:
        # Connect to output database file (by convention) + init
        output_databasename = 'main'
        conn = sqlite3.connect(output_path)
        conn.enable_load_extension(True)
        conn.load_extension('mod_spatialite')
        conn.execute('PRAGMA cache_size = 10000;')  # Number of cache pages (page = 4096 bytes)
        conn.execute('PRAGMA temp_store = MEMORY;')
        if journal_mode is not None:
            conn.execute(f"PRAGMA journal_mode = {journal_mode};")
        
        # Init as gpkg
        conn.execute('SELECT gpkgCreateBaseTables();')
        conn.execute('SELECT EnableGpkgMode();')
        if crs_epsg != -1:
            conn.execute(f"SELECT gpkgInsertEpsgSRID({crs_epsg})")

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
        
        #sql_stmt = f'CREATE TABLE {output_databasename}."{output_layer}" AS\n{sql_stmt}' 
        columns_for_create = [f'"{column[1]}" {column[2]}\n' for column in columns if column[1] != 'geom']
        sql = f'CREATE TABLE {output_databasename}."{output_layer}" ({", ".join(columns_for_create)})'
        conn.execute(sql)

        # Add geometry column
        conn.execute(f"SELECT gpkgAddGeometryColumn('{output_layer}', 'geom', '{output_geometrytype.name}', 0, 0, {crs_epsg});")
        
        columns_for_insert = [f'"{column[1]}"' for column in columns]
        sql = f'INSERT INTO {output_databasename}."{output_layer}" ({", ".join(columns_for_insert)})\n{sql_stmt}' 
        conn.execute(sql)
        cur.execute(f"SELECT gpkgAddGeometryTriggers('{output_layer}', 'geom');")
        
        # Create spatial index
        if create_spatial_index is True:
            cur.execute(f"SELECT UpdateLayerStatistics('{output_layer}', 'geom');")
            cur.execute(f"SELECT gpkgAddSpatialIndex('{output_layer}', 'geom');")
        conn.commit()

        logger.info('End')
    except Exception as ex:
        raise Exception(f"Error executing {sql}") from ex

if __name__ == '__main__':
    None
