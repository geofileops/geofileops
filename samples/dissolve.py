
import json
import logging
import logging.config
from pathlib import Path
import sys
[sys.path.append(i) for i in ['.', '..']]

from geofileops import geofileops

if __name__ == '__main__':

    # Init logging
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / 'logging.json', 'r') as log_config_file:
        log_config_dict = json.load(log_config_file)
    logging.config.dictConfig(log_config_dict)
    logging.raiseExceptions = True
    logger = logging.getLogger()
    
    # General init
    input_layer = None
    groupby_columns = None
    aggfunc = None
    output_layer = None
    explodecollections = False

    # Init input files
    input_cardsheets_path = r"X:\GIS\GIS DATA\Versnijdingen\Kaartbladversnijdingen_NGI_numerieke_reeks_Shapefile\Shapefile\Kbl8.shp"
    
    #input_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_14\Prc_2018_bufm1_sealed_14_inter.gpkg"
    #output_path = r"X:\Monitoring\OrthoSeg\sealedsurfaces\output_vector\sealedsurfaces_14\Prc_2018_bufm1_sealed_14_inter_diss.gpkg"
    #groupby_columns = ['l2_CODE_OBJ', 'l2_GWSCOD_H', 'l2_GESP_PM']
    #input_layer = None
    #output_layer = None
    
    # Collect: 6u20

    #input_path = r"c:\temp\BRUGIS01_collect_pierog.gpkg"
    #output_path = r"c:\temp\BRUGIS01_collect_union_pierog.gpkg"
    
    #input_path=r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03_50m_buffer_diss_card_gpd_clip.gpkg"
    #output_path=r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03_50m_buffer_diss_card_gpd_clip_diss_gpd.gpkg"
    
    input_path = r"X:\PerPersoon\PIEROG\Tmp\prc_2019.gpkg"
    output_path = r"X:\PerPersoon\PIEROG\Tmp\prc_2019_diss_nogroup.gpkg"
    #groupby_columns = ['GEWASGROEP']
    #aggfunc = 'first'
    explodecollections = True
    output_layer = None

    # Go!
    logger.info("Start")
    geofileops.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=groupby_columns,
            aggfunc=aggfunc,
            explodecollections=explodecollections,
            input_layer=input_layer,
            output_layer=output_layer,
            #bbox=(100000, 200000, 105000, 205000),
            input_cardsheets_path=input_cardsheets_path,
            force=True)
    logger.info("Ready")
