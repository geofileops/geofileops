
import json
import logging
import logging.config
import os
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

import geofileops.geofileops as geofileops

if __name__ == '__main__':

    ##### Init #####
    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()

    # 
    output_path = None
    output_filename = None

    # GRB tuinen
    #input_to_select_from_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\adp_parcels_repaired.gpkg"
    #input_to_compare_with_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\gbg_repaired.gpkg"
    #output_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\adp_parcels_repaired_INTERSECTING_gbg_repaired.gpkg"
    
    # Tuinen
    #input_to_select_from_path = r"X:\Monitoring\OrthoSeg\Prc_2018_bufm1.gpkg"
    #input_to_compare_with_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-05-20_QA Ingesloten tuin 2019 met nieuw algoritme\02_Shape\pot_tuin.shp"
    #output_filename = f"{input1_filename_noext}_INTERSECTING_pot_tuin.gpkg"

    # Prc Denemarken
    input_to_select_from_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\prc_DK2019.gpkg"
    input_to_compare_with_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\DK_2019\sealedsurfaces_23_0_523_DK_2019.gpkg"

    input1_dir, input1_filename = os.path.split(input_to_select_from_path)
    input2_dir, input2_filename = os.path.split(input_to_compare_with_path)
    input1_filename_noext, _ = os.path.splitext(input1_filename)
    input2_filename_noext, _ = os.path.splitext(input2_filename)

    output_dir = input1_dir
    if output_path is None:
        if output_filename is None:
            output_filename = f"{input1_filename_noext}_INTERSECTING_{input2_filename_noext}.gpkg"
        output_path = os.path.join(output_dir, output_filename)

    ##### Go! #####
    logger.info("Start")
    geofileops.export_by_location(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path,
            force=True,
            verbose=False)
    logger.info("Ready")
