
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
    input_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_14\Prc_2018_bufm1_sealed_14_inter.gpkg"
    output_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_14\Prc_2018_bufm1_sealed_14_inter_diss.gpkg"
    groupby_columns = ['l2_CODE_OBJ', 'l2_GWSCOD_H', 'l2_GESP_PM']
    input_layer = None
    output_layer = None
    '''

    # Collect: 6u20
    input_layer = None
    groupby_columns = None
    output_layer = None

    '''    
    input_path = r"c:\temp\BRUGIS01.gpkg"
    input_layer = "GEBOUWENREPAIRED"    
    output_path = r"c:\temp\BRUGIS01_collect_pierog.gpkg"
    '''
    input_path = r"c:\temp\BRUGIS01_collect_pierog.gpkg"
    output_path = r"c:\temp\BRUGIS01_collect_union_pierog.gpkg"

    # Go!
    logger.info("Start")
    geofile_ops.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=groupby_columns,
            input_layer=input_layer,
            output_layer=output_layer,
            force=True)
    logger.info("Ready")
