
import json
import logging
import logging.config
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

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
    input_to_select_from_path = Path(r"X:\GIS\GIS DATA\Percelen_ALP\Vlaanderen\Perc_VL_2020_2020-05-25\perc_2020_met_k_2020-05-25.gpkg")
    input_to_compare_with_path = Path(r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\BEFL-2020-ofw\sealedsurfaces_27_136_BEFL-2020-ofw.gpkg")

    # Prepare output path
    if output_path is None:
        if output_filename is None:
            output_filename = f"{input_to_select_from_path.stem}_INTERSECTING_{input_to_compare_with_path.stem}.gpkg"
        output_path = input_to_select_from_path.parent / output_filename

    ##### Go! #####
    logger.info("Start")
    geofileops.export_by_location(
            input_to_select_from_path=input_to_select_from_path,
            input_to_compare_with_path=input_to_compare_with_path,
            output_path=output_path,
            input1_columns=['CODE_OBJ'],
            force=True,
            verbose=True)
    logger.info("Ready")
