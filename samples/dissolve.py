
import json
import logging
import logging.config
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

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
    aggfunc = 'first'
    output_layer = None
    tiles_path = None
    explodecollections = True
    clip_on_tiles = False
    output_layer = None

    # Init input files
    #input_cardsheets_path = r"X:\GIS\GIS DATA\Versnijdingen\Kaartbladversnijdingen_NGI_numerieke_reeks_Shapefile\Shapefile\Kbl8.shp"
    
    #input_path = Path(r"X:\Monitoring\OrthoSeg\trees\output_vector\BEFL-2019\trees_04_179_BEFL-2019.gpkg")
    input_path = Path(r"X:\Monitoring\OrthoSeg\trees\output_vector\BEFL-2019\trees_05_472_BEFL-2019.gpkg")
    output_path = input_path.parent / f"{input_path.stem }_dissolvetest{input_path.suffix}"
    #groupby_columns = ['l2_CODE_OBJ', 'l2_GWSCOD_H', 'l2_GESP_PM']
    #explodecollections = True
    #keep_cardsheets = False
    
    # Collect: 6u20

    #input_path = r"c:\temp\BRUGIS01_collect_pierog.gpkg"
    #output_path = r"c:\temp\BRUGIS01_collect_union_pierog.gpkg"
    
    #input_path=r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03_50m_buffer_diss_card_gpd_clip.gpkg"
    #output_path=r"X:\__IT_TEAM_ANG_GIS\Taken\2020\2020-04-09_FasterDissolve\GBG_woningen03_50m_buffer_diss_card_gpd_clip_diss_gpd.gpkg"
    
    #input_path = Path(r"X:\Monitoring\GeofileOps\test_water_WVLA.gpkg")
    #output_path = input_path.parent / f"{input_path.stem }_dissolvetest{input_path.suffix}"
    #groupby_columns = ['GEWASGROEP']
    #aggfunc = 'first'

    # Go!
    from fiona import supported_drivers

    logger.info("Start")
    geofileops.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=groupby_columns,
            aggfunc=aggfunc,
            explodecollections=explodecollections,
            clip_on_tiles=clip_on_tiles,
            tiles_path=tiles_path,
            input_layer=input_layer,
            output_layer=output_layer,
            #nb_parallel=1,
            verbose=True,
            force=True)
    logger.info("Ready")
