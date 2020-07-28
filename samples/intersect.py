
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

    # Use OSGeo4W for ogr operations
    if 'GDAL_BIN' not in os.environ:
        os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
        os.environ['PATH'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin;" + os.environ['PATH']

    input1_path = Path(r"X:\PerPersoon\PIEROG\Taken\2020\2020-07-23_rbh2010vsrbh2019\rbh2019_ms.gpkg")
    input2_path = Path(r"X:\PerPersoon\PIEROG\Taken\2020\2020-07-23_rbh2010vsrbh2019\rbh2010_ms.gpkg")

    output_filename = f"{input1_path.stem}_INTER_{input2_path.stem}.gpkg"
    output_path = input1_path.parent / output_filename

    ##### Go! #####
    logger.info("Start")
    geofileops.intersect(
            input1_path=input1_path,
            input2_path=input2_path,
            output_path=output_path,
            force=True)
    logger.info("Ready")
