
import json
import logging
import os
import sys
[sys.path.append(i) for i in ['.', '..']]

import geofile_ops.geofile_ops as geofile_ops

if __name__ == '__main__':

    ##### Init #####
    # Init logging
    #logging.config.fileConfig('bin/logging.ini')
    with open('bin/logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()

    # 
    input_to_select_from_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\adp_parcels_repaired.gpkg"
    input_to_compare_with_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\gbg_repaired.gpkg"
    output_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\adp_parcels_repaired_INTERSECTING_gbg_repaired.gpkg"
    
    input_to_select_from_path = r"X:\Monitoring\OrthoSeg\Prc_2018_bufm1.gpkg"
    input_to_compare_with_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\pot_tuin.shp"
    
    input1_dir, input1_filename = os.path.split(input_to_select_from_path)
    input2_dir, input2_filename = os.path.split(input_to_compare_with_path)
    input1_filename_noext, _ = os.path.splitext(input1_filename)
    input2_filename_noext, _ = os.path.splitext(input2_filename)
    
    output_dir = input1_dir
    #output_filename = f"{input1_filename_noext}_INTERSECTING_{input2_filename_noext}.gpkg"
    output_filename = f"{input1_filename_noext}_INTERSECTING_pot_tuin.gpkg"
    output_path = os.path.join(output_dir, output_filename)

    ##### Go! #####
    logger.info("Start")
    geofile_ops.export_by_location(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path,
            force=True)
    logger.info("Ready")
