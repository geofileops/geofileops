import json
import logging
import logging.config
import os
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

import geofileops.geofileops as geofileops

def main():
    
    ##### Init #####
    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()
    
    # Prepare output path
    tolerance = 0.20
    input_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_BEFL_2019_ofw_16\sealedsurfaces_BEFL_2019_ofw_16_orig.gpkg"
    input_path_noext, ext = os.path.splitext(input_path)
    output_path = f"{input_path_noext}_simpl_{str(tolerance).replace('.', '')}{ext}"
    
    ##### Go! #####
    logger.info("Start")
    geofileops.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=tolerance,
            force=True)
    logger.info("Ready")

if __name__ == '__main__':
    main()
