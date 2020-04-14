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
    input_path = r"X:\Monitoring\OrthoSeg\christmastrees\input_labels\christmastrees_labellocations.gpkg"
    geofile_ops.rename_layer(
            path=input_path,
            layer='topobuildings_labellocations',
            new_layer='christmastrees_labellocations')

if __name__ == '__main__':
    main()
