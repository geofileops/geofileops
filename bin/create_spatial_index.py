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
    path=r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_BEFL_2019_ofw_18\sealedsurfaces_BEFL_2019_ofw_18.gpkg"
    geofile_ops.create_spatial_index(path=path, layer=None)

if __name__ == '__main__':
    main()
