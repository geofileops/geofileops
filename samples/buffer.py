import json
import logging
import sys
[sys.path.append(i) for i in ['.', '..']]

import geofile_ops.geofile_ops as geofile_ops

def main():
    
    # Init logging
    #logging.config.fileConfig('bin/logging.ini')
    with open('bin/logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()
    
    # Go!
    logger.info("Start")
    geofile_ops.buffer(
            input_path=r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_14\Prc_2018_bufm1_sealed_14_inter.gpkg",
            output_path=r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_14\Prc_2018_bufm1_sealed_14_inter_bufp1.gpkg",
            buffer=1,
            force=True)
    logger.info("Ready")

if __name__ == '__main__':
    main()
