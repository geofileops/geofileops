import json
import logging
import logging.config
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

from geofileops import geofile

def main():
    
    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    #logger = logging.getLogger()
    
    # Go!
    input_path = r"X:\Monitoring\OrthoSeg\christmastrees\input_labels\christmastrees_labellocations.gpkg"
    geofile.rename_layer(
            path=input_path,
            layer='topobuildings_labellocations',
            new_layer='christmastrees_labellocations')

if __name__ == '__main__':
    main()
