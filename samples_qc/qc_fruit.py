
import json
import logging
import logging.config
import os
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

from geofileops import geofileops
from geofileops import geofile

if __name__ == '__main__':

    ##### Init #####
    # Input files
    fruit_path = r"X:\Monitoring\OrthoSeg\fruit\output_vector\fruit_07\fruit_07_simpl_shap.gpkg"
    prc_path = r"X:\GIS\GIS DATA\Percelen_ALP\Vlaanderen\Perc_VL_2019_2019-08-27\perc_2019_met_k_2019-08-27.shp"
    gbg_path = r""
    adp_path = r""

    # Dir with intemediary files
    working_dir = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\1-Tussentijdsefiles"
    log_dir = os.path.join(working_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    # TODO: seperate log file per run/date/???
    log_config_dict['handlers']['file']['filename'] = os.path.join(log_dir, 'logfile.log')
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()

    verbose = True

    ##### Processing! #####

    ### Voorbereiding basislagen ###
    # -1 buffer op percelen
    prc_bufm1 = os.path.join(working_dir, '010_prc_2019_2019-08-27_bufm1.gpkg')
    geofileops.buffer(
            input_path=prc_path,
            output_path=prc_bufm1,
            buffer=-1,
            verbose=verbose)

    # alleen percelen waarin geen tuin mag voorkomen: filter:
    # Akkerbouw percelen gaven alleen valse positieven!
    '''
    sqlite_stmt = """
            SELECT * FROM \"010_prc_2019_2019-08-27_bufm1\"
             WHERE gwscod_h NOT IN ('1','2')
               AND (gesp_pm IS null 
                    OR (gesp_pm NOT LIKE '%SER%'
                        AND gesp_pm NOT LIKE '%SGM%'
                        AND gesp_pm NOT LIKE '%LOO%'))
            """
    '''
    sqlite_stmt = """
            SELECT * FROM \"010_prc_2019_2019-08-27_bufm1\"
             WHERE gwscod_h = '60'
             """

    prc_bufm1_filtered = os.path.join(working_dir, '020_prc_2019_bufm1_filtered.gpkg')
    geofileops.select(
            input_path=prc_bufm1,
            output_path=prc_bufm1_filtered,
            sqlite_stmt=sqlite_stmt,
            verbose=verbose)

    ### Voorbereiding gbg ###
    # alleen gebouwen die groot genoeg zijn
    sqlite_stmt = """SELECT * 
                       FROM gbg
                      WHERE oppervl > 50
                        AND lbltype <> 'bijgebouw' """
    gbg_filtered_path = os.path.join(working_dir, '040_gbg_filtered.gpkg')
    geofileops.select(
            input_path=gbg_path,
            output_path=gbg_filtered_path,
            sqlite_stmt=sqlite_stmt,
            verbose=verbose)

    ### Voorbereiding adp ###
    # Alleen adp in de buurt van percelen
    adp_bijprc_path = os.path.join(working_dir, '050_adp_bijprc.gpkg')
    geofileops.export_by_location(
        input_to_select_from_path=adp_path,
        input_to_compare_with_path=prc_bufm1_filtered,
        output_path=adp_bijprc_path,
            verbose=verbose)
    
    # Alleen adp percelen die gebouw in zich hebben
    adp_bijprc_bijgbg_path = os.path.join(working_dir, '060_adp_bijprc_bijgbg.gpkg')
    geofileops.export_by_location(
            input_to_select_from_path=adp_bijprc_path,
            input_to_compare_with_path=gbg_filtered_path,
            output_path=adp_bijprc_bijgbg_path,
            verbose=verbose)

    # Verder uitfilteren resultaat
    sqlite_stmt = """SELECT * 
                       FROM \"060_adp_bijprc_bijgbg\"
                      WHERE l1_l1_oppervl < 3000
                        AND area_inters > 20"""
    adp_pot_tuin_path = os.path.join(working_dir, '070_pot_tuin.gpkg')
    geofileops.select(
            input_path=adp_bijprc_bijgbg_path,
            output_path=adp_pot_tuin_path,
            sqlite_stmt=sqlite_stmt,
            verbose=verbose)

    # Bepalen stukken tuin in percelen
    prc_bufm1_filtered_INTERS_pot_tuin = os.path.join(
            working_dir, '080_prc_2019_bufm1_filtered_INTERS_pot_tuin.gpkg')
    geofileops.intersect(
            input1_path=prc_bufm1_filtered,
            input2_path=adp_pot_tuin_path,
            output_path=prc_bufm1_filtered_INTERS_pot_tuin,
            verbose=verbose)
    
    # Aggregeren per perceel
    prc_bufm1_tuin_path = os.path.join(
            working_dir, '090_prc_2019_bufm1_tuin.gpkg')
    geofileops.dissolve(
            input_path=prc_bufm1_filtered_INTERS_pot_tuin,
            output_path=prc_bufm1_tuin_path,
            groupby_columns=["l1_CODE_OBJ"],
            verbose=verbose)
    geofile.add_column(
            path=prc_bufm1_tuin_path,
            column_name='area')

    # Verder uitfilteren resultaat
    sqlite_stmt = """SELECT * 
                       FROM \"090_prc_2019_bufm1_tuin\"
                      WHERE area > 25"""
    prc_bufm1_tuin_filtered_path = os.path.join(working_dir, '100_prc_2019_bufm1_tuin_filtered.gpkg')
    geofileops.select(
            input_path=prc_bufm1_tuin_path,
            output_path=prc_bufm1_tuin_filtered_path,
            sqlite_stmt=sqlite_stmt,
            verbose=verbose)

    # Alleen overlappingen die dicht genoeg bij gebouw liggen
    prc_bufm1_tuin_filtered_bijgbg_path = os.path.join(working_dir, '110_prc_2019_bufm1_tuin_fil_bijgbg.gpkg')
    geofileops.export_by_distance(
            input_to_select_from_path=prc_bufm1_tuin_filtered_path,
            input_to_compare_with_path=gbg_filtered_path,
            output_path=prc_bufm1_tuin_filtered_bijgbg_path,
            max_distance=30,
            verbose=verbose)

    # Adp percelen die met de stukken tuin overlappen
    adp_bijtuin_path = os.path.join(working_dir, '120_adp_bijtuin.gpkg')
    geofileops.export_by_location(
            input_to_select_from_path=adp_pot_tuin_path,
            input_to_compare_with_path=prc_bufm1_tuin_filtered_bijgbg_path,
            output_path=adp_bijtuin_path,
            verbose=verbose)
