import json
import logging
import logging.config
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

import geofile_ops.geofile_ops as geofile_ops

def main():
    
    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()
    
    input_path = Path(r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03.gpkg")
    output_path = input_path.parent / f"{input_path.stem}_buf{input_path.suffix}"

    # Go!
    logger.info("Start buffer")
    geofile_ops.buffer(
            input_path=str(input_path),
            output_path=str(output_path),
            buffer=1,
            force=True)
    logger.info("Ready buffer")

if __name__ == '__main__':
    main()
