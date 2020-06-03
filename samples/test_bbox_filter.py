
import time

import geopandas as gpd
from geofileops import geofile

def main():

    input_path = r"X:\GIS\GIS DATA\Gewesten\2005\Belgie\BELGIE.shp"
    input_path = r"X:\PerPersoon\PIEROG\Tmp\belgie.gpkg"
    input_path = r"X:\GIS\GIS DATA\Percelen_ALP\Vlaanderen_geopunt\Landbouwgebruikspercelen_LV_2019_GewVLA_Shapefile\Shapefile\Lbgbrprc18.shp"

    # Read all records that intersect the bbox
    bbox = (180000, 180000, 185000, 185000)
    bbox = (0, 0, 300000, 300000)
    input_gdf = geofile.read_file(filepath=input_path, bbox=bbox)

    # Get representative_point of each polygon
    '''
    t = time.process_time()
    input_gdf['representative_point'] = input_gdf.geometry.representative_point()
    elapsed_time = time.process_time() - t 
    print(f"Calculate representative_point of each polygon took {elapsed_time}")
    print(f"Number items before filter: {len(input_gdf)}")
    t = time.process_time()
    curr_geometry_column = input_gdf.geometry.name
    input_gdf.set_geometry('representative_point', inplace=True)

    t = time.process_time()
    input_gdf = input_gdf.loc[
        (input_gdf.geometry.x >= bbox[0]) &
        (input_gdf.geometry.y >= bbox[1]) &
        (input_gdf.geometry.x < bbox[2]) &
        (input_gdf.geometry.y < bbox[3])] 
    elapsed_time = time.process_time() - t 
    print(f"Filter took {elapsed_time}")
    print(f"Number items after filter: {len(input_gdf)}")

    input_gdf.set_geometry(curr_geometry_column, inplace=True)
    '''

    t = time.process_time()
    representative_point_gs = input_gdf.geometry.representative_point()
    input_gdf['representative_point_x'] = representative_point_gs.x
    input_gdf['representative_point_y'] = representative_point_gs.y
    elapsed_time = time.process_time() - t 
    print(f"Calculate representative_point of each polygon took {elapsed_time}")
    print(f"Number items before filter: {len(input_gdf)}")

    t = time.process_time()
    input_gdf = input_gdf.loc[
        (input_gdf['representative_point_x'] >= bbox[0]) &
        (input_gdf['representative_point_y'] >= bbox[1]) &
        (input_gdf['representative_point_x'] < bbox[2]) &
        (input_gdf['representative_point_y'] < bbox[3])] 
    elapsed_time = time.process_time() - t 
    print(f"Filter took {elapsed_time}")
    print(f"Number items after filter: {len(input_gdf)}")

    input_gdf.drop(['representative_point_x', 'representative_point_y'], axis=1, inplace=True)
if __name__ == '__main__':
    main()
