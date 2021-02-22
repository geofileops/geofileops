# -*- coding: utf-8 -*-
"""
Module containing utilities regarding low level vector operations.
"""

import logging

import geopandas as gpd

from . import geometry

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# GeoDataFrame helpers
#-------------------------------------------------------------

def polygons_to_lines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cardsheets_lines = []
    for cardsheet_poly in gdf.itertuples():
        cardsheet_boundary = cardsheet_poly.geometry.boundary
        if cardsheet_boundary.type == 'MultiLineString':
            for line in cardsheet_boundary:
                cardsheets_lines.append(line)
        else:
            cardsheets_lines.append(cardsheet_boundary)

    cardsheets_lines_gdf = gpd.GeoDataFrame(geometry=cardsheets_lines, crs=gdf.crs)

    return cardsheets_lines_gdf
