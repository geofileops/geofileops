
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
    
    input_path=r"X:\Monitoring\OrthoSeg\topobuildings\output_vector\topobuildings_07_BEFL_topo_1969\topobuildings_07_BEFL_topo_1969.gpkg"
    input_cardsheets_path=r"X:\GIS\GIS DATA\Versnijdingen\Kaartbladversnijdingen_NGI_numerieke_reeks_Shapefile\Shapefile\Kbl16.shp"
    output_path=r"X:\Monitoring\OrthoSeg\topobuildings\output_vector\topobuildings_07_BEFL_topo_1969\topobuildings_07_BEFL_topo_1969_diss_card.gpkg"
            
    # Go!
    logger.info("Start")
    geofile_ops.dissolve_cardsheets(
            input_path=input_path,
            input_cardsheets_path=input_cardsheets_path,
            output_path=output_path,
            groupby_columns=None,
            explodecollections=True,
            #groupby_columns=['l2_CODE_OBJ', 'l2_GWSCOD_H', 'l2_GESP_PM'],
            verbose=True,
            force=True)
    logger.info("Ready")
