
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

    # sealedsurfaces 2018
    input1_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_BEFL_2018_16\sealedsurfaces_BEFL_2018_16_orig.gpkg"
    input2_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_13\Prc_2018_bufm1.gpkg"
    
    # greenhouses 2019
    input1_path = r"X:\Monitoring\OrthoSeg\greenhouses\output_vector\greenhouses_BEFL_2019_ofw_36\greenhouses_BEFL_2019_ofw_36_simpl_shap.gpkg"
    input2_path = r"X:/__IT_TEAM_ANG_GIS/Taken/2019/2019-08-28_QA_Serres/Prc_2019_2019-08-27_bufm1.gpkg"

    # sealedsurfaces 2019
    input1_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_BEFL_2019_ofw_20\sealedsurfaces_BEFL_2019_ofw_20.gpkg"
    input2_path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-08-28_QA_Serres\1-Tussentijdse_files\Prc_2019_2019-08-27_bufm1.gpkg"

    input1_dir, input1_filename = os.path.split(input1_path)
    input2_dir, input2_filename = os.path.split(input2_path)
    input1_filename_noext, _ = os.path.splitext(input1_filename)
    input2_filename_noext, _ = os.path.splitext(input2_filename)
    
    output_dir = input1_dir
    output_filename = f"{input1_filename_noext}_INTER_{input2_filename_noext}.gpkg"
    output_path = os.path.join(output_dir, output_filename)

    ##### Go! #####
    logger.info("Start")
    geofile_ops.intersect(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            force=True)
    logger.info("Ready")
