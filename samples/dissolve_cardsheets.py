
import json
import logging
import sys
[sys.path.append(i) for i in ['.', '..']]

import geofile_ops.geofile_ops as geofile_ops

if __name__ == '__main__':

    # Init logging
    #logging.config.fileConfig('bin/logging.ini')
    with open('bin/logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logging.raiseExceptions = True
    logger = logging.getLogger()
    
    '''
    input_path=r"X:\Monitoring\OrthoSeg\topobuildings\output_vector\topobuildings_07_BEFL_topo_1969\topobuildings_07_BEFL_topo_1969.gpkg"
    output_path=r"X:\Monitoring\OrthoSeg\topobuildings\output_vector\topobuildings_07_BEFL_topo_1969\topobuildings_07_BEFL_topo_1969_diss_card.gpkg"
    '''
    input_path=r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03_50m_buffer.gpkg"
    output_path=r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03_50m_buffer_diss_card_gpd_clip.gpkg"

    input_cardsheets_path=r"X:\GIS\GIS DATA\Versnijdingen\Kaartbladversnijdingen_NGI_numerieke_reeks_Shapefile\Shapefile\Kbl8.shp"

    # Go!
    logger.info("Start")
    geofile_ops.dissolve_cardsheets_gpd(
            input_path=input_path,
            input_cardsheets_path=input_cardsheets_path,
            output_path=output_path,
            groupby_columns=None,
            explodecollections=True,
            #groupby_columns=['l2_CODE_OBJ', 'l2_GWSCOD_H', 'l2_GESP_PM'],
            verbose=True,
            force=True)
    logger.info("Ready")
