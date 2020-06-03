
import json
import logging
import logging.config
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

from geofileops import geofileops

if __name__ == '__main__':

    # Init
    input_cardsheets_path = r"X:\GIS\GIS DATA\Versnijdingen\Kaartbladversnijdingen_NGI_numerieke_reeks_Shapefile\Shapefile\Kbl8.shp"
    input_path = Path(r"X:\Monitoring\OrthoSeg\topobuildings\output_vector\BEFL-topo-1989\topobuildings_17_361_BEFL-topo-1989.gpkg")
    #input_path=r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03_50m_buffer.gpkg"

    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    # If the filename in the file handler is not an absolute path... log in the input dir 
    if('handlers' in log_config_dict 
       and 'filename' in log_config_dict['handlers']
       and Path(log_config_dict['handlers']['filename']).is_absolute() is False):
        log_config_dict['handlers']['filename'] = str(input_path.parent / log_config_dict['handlers']['filename'])
    logging.config.dictConfig(log_config_dict)
    logging.raiseExceptions = True
    logger = logging.getLogger()
    
    # Prepare output path
    output_path = input_path.parent / f"{input_path.stem}_diss_card{input_path.suffix}"

    # Go!
    logger.info("Start")
    geofileops.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=None,
            #groupby_columns=['l2_CODE_OBJ', 'l2_GWSCOD_H', 'l2_GESP_PM'],
            explodecollections=True,
            input_cardsheets_path=input_cardsheets_path,
            keep_cardsheets=True,
            verbose=False,
            force=True)
    logger.info("Ready")
