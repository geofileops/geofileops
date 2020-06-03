import json
import logging
import logging.config
from pathlib import Path
import pprint
import sys
[sys.path.append(i) for i in ['.', '..']]

from geofileops import geofile

def main():
    
    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()
    
    # Go!
    path = r"X:\__IT_TEAM_ANG_GIS\Taken\2019\2019-08-28_QA_Serres\1-Tussentijdse_files\Prc_2019_2019-08-27_bufm1.gpkg"
    result = geofile.getfileinfo(path=path, verbose=False)  
    logger.info(f"Result: {pprint.pformat(result)}")

if __name__ == '__main__':
    main()
