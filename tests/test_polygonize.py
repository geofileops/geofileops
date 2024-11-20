import geopandas as gpd
import pytest
from osgeo import gdal
from shapely import box

import geofileops as gfo
from tests.test_helper import assert_geodataframe_equal

gdal.UseExceptions()


@pytest.mark.parametrize("nb_tiles", [1, 2])
def test_polygonize(tmp_path, nb_tiles):
    """Test polygonize for a simple case both with and without tiling."""
    # Prepare base vector
    input_vector_path = tmp_path / "input.gpkg"
    input_gdf = gpd.GeoDataFrame(
        data={"DN": [1, 2]}, geometry=[box(0, 0, 5, 5), box(10, 0, 15, 5)], crs=31370
    )
    input_gdf.to_file(input_vector_path)

    # Prepare input raster
    input_raster_path = tmp_path / "input.tif"
    options = gdal.RasterizeOptions(
        creationOptions=["COMPRESS=DEFLATE"],
        outputType=gdal.GDT_Int32,
        outputBounds=input_gdf.total_bounds,
        xRes=1,
        yRes=1,
        targetAlignedPixels=True,
        attribute="DN",
    )
    gdal.Rasterize(
        srcDS=input_vector_path, destNameOrDestDS=input_raster_path, options=options
    )

    # Determine max_tile_size_mb to get the desired number of tiles
    if nb_tiles == 1:
        max_tile_size_mb = 500
    else:
        nb_pixels = box(*input_gdf.total_bounds).area
        max_tile_size_mb = (nb_pixels // nb_tiles) / 1024 / 1024

    # Test
    output_path = tmp_path / "output.gpkg"
    gfo.polygonize(
        input_path=input_raster_path,
        output_path=output_path,
        simplify_tolerance=1,
        dissolve_result=False,
        max_tile_size_mb=max_tile_size_mb,
    )

    # Validate
    output_gdf = gpd.read_file(output_path)

    # Ignore the background polygon in the output
    output_gdf = output_gdf[output_gdf["DN"] != 0]

    assert_geodataframe_equal(
        output_gdf, input_gdf, sort_values=True, check_dtype=False, normalize=True
    )
