import json
import logging
import logging.config
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofileops

if __name__ == '__main__':

    # Use OSGeo4W for ogr operations
    if 'GDAL_BIN' not in os.environ:
        os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
        os.environ['PATH'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin;" + os.environ['PATH']

    ##### Init #####
    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()
    
    # Input
    force = False
    verbose = True
    input_prc_path = Path(r"X:\GIS\GIS DATA\Percelen_ALP\Vlaanderen\Perc_VL_2020_2020-05-25\perc_2020_met_k_2020-05-25.gpkg")
    
    #input_sealedsurfaces_path = Path(r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\BEFL-2020-ofw\sealedsurfaces_27_136_BEFL-2020-ofw.gpkg")
    #output_dir = Path(r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-07-20_QA_verharding\02-Tussenfiles")
    
    input_compare_to_path = Path(r"X:\Monitoring\OrthoSeg\horsetracks\output_vector\BEFL-2020-ofw\horsetracks_22_206_BEFL-2020-ofw_simpl.gpkg")
    output_dir = Path(r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-07-27_QA_Paardenhouders\02-Tussenfiles")
    
    # First apply negative buffer 
    prc_m1m_path = output_dir / 'prc_m1m.gpkg'
    geofileops.buffer(
            input_path=input_prc_path,
            output_path=prc_m1m_path,
            distance=-1,
            columns=['geometry', 'CODE_OBJ', 'GWSCOD_H', 'GESP_PM'],
            force=force)
    
    # Now calculate intersection with other layer 
    prc_m1m_inters_compare_to_path = (
            prc_m1m_path.parent / f"{prc_m1m_path.stem}_INTERS_{input_compare_to_path.stem}.gpkg")
    geofileops.intersect(
            input1_path=prc_m1m_path,
            input2_path=input_compare_to_path,
            output_path=prc_m1m_inters_compare_to_path,
            explodecollections=True,
            force=force)

    # Only keep interections > 
    prc_m1m_inters_compare_to_filter_path = (
            prc_m1m_inters_compare_to_path.parent / f"{prc_m1m_inters_compare_to_path.stem}_filter.gpkg")
    sql_stmt = f"""
            SELECT ST_Multi(ST_union(geom)) AS geom
                  ,l1_code_obj
                  ,l1_gwscod_h
                  ,l1_gesp_pm
                  ,area_inter
              FROM "{prc_m1m_inters_compare_to_path.stem}"
             WHERE (area_inter >= 100 OR ST_area(geom) > 25)
               AND l1_GWSCOD_H not in ('1','2','6','7','8','9','81','85','99','100','895','962','999','9536','9573','9574','9823','9825','9829')
               AND (l1_GESP_PM is null
                    OR (l1_GESP_PM not like '%SER%'
                        AND l1_GESP_PM not like '%SGM%'
                        AND l1_GESP_PM not like '%PLA%'
                        AND l1_GESP_PM not like '%NPO%'
                        AND l1_GESP_PM not like '%CON%'
                        AND l1_GESP_PM not like '%CIV%'))
              GROUP BY l1_code_obj, l1_gwscod_h, l1_gesp_pm, area_inter
            """
    geofileops.select(
            input_path=prc_m1m_inters_compare_to_path,
            output_path=prc_m1m_inters_compare_to_filter_path,
            sql_stmt=sql_stmt,
            verbose=verbose,
            force=force)
        
    '''
    # Select parcels that intersect with sealedsurfaces
    prc_m1m_inters_sealeds_path = (
            prc_m1m_path.parent / f"{prc_m1m_path.stem}_INTERS_{input_sealedsurfaces_path.stem}.gpkg")

    geofileops.export_by_location(
            input_to_select_from_path=prc_m1m_path,
            input_to_compare_with_path=input_sealedsurfaces_path,
            output_path=prc_m1m_inters_sealeds_path,
            verbose=verbose,
            force=force)
    
    prc_m1m_inters_sealeds_filter_path = (
            prc_m1m_inters_sealeds_path.parent / f"{prc_m1m_inters_sealeds_path.stem}_filter.gpkg")
    sql_stmt = f"""
            SELECT geom
                  ,l1_code_obj
                  ,l1_gwscod_h
                  ,l1_gesp_pm
                  ,area_inters
              FROM "{prc_m1m_inters_sealeds_path.stem}"
             WHERE area_inters >= 25
               AND l1_GWSCOD_H not in ('1','2','6','7','8','9','81','85','99','100','895','962','999','9536','9573','9574','9823','9825','9829')
               AND (l1_GESP_PM is null
                    OR (l1_GESP_PM not like '%SER%'
                        AND l1_GESP_PM not like '%SGM%'
                        AND l1_GESP_PM not like '%PLA%'
                        AND l1_GESP_PM not like '%NPO%'
                        AND l1_GESP_PM not like '%CON%'
                        AND l1_GESP_PM not like '%CIV%'))
            """
    geofileops.select(
            input_path=prc_m1m_inters_sealeds_path,
            output_path=prc_m1m_inters_sealeds_filter_path,
            sql_stmt=sql_stmt,
            verbose=verbose,
            force=force)
    '''
