import shapely

SHAPELY_GE_20 = str(shapely.__version__).split(".")[0] >= "2"
