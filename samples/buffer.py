import json
import logging
import logging.config
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofileops

def main():
    
    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logger = logging.getLogger()
    
    input_path = Path(r"X:\GIS\GIS DATA\Percelen_ALP\Vlaanderen\Perc_VL_2020_2020-10-05\perc_2020_met_k_2020-10-05.gpkg")
    output_path = input_path.parent / f"{input_path.stem}_buf{input_path.suffix}"

    # Go!
    logger.info("Start buffer")
    geofileops.buffer(
            input_path=str(input_path),
            output_path=str(output_path),
            distance=-1,
            verbose=True)
    logger.info("Ready buffer")

if __name__ == '__main__':
    main()
