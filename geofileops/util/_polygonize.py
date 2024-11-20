"""Polygonize a raster file to a vector file, tiled."""

import logging
import multiprocessing
import shutil
import time
from concurrent import futures
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import geopandas as gpd
import pygeoops
import pyproj
import shapely
from osgeo import gdal
from osgeo_utils import gdal_polygonize

from geofileops import fileops, geoops
from geofileops.helpers._configoptions_helper import ConfigOptions
from geofileops.util import _general_util, _io_util, _processing_util
from geofileops.util._general_util import LeaveTry
from geofileops.util._geofileinfo import GeofileInfo
from geofileops.util._geometry_util import SimplifyAlgorithm

if TYPE_CHECKING:
    import os
    from typing import Any

gdal.UseExceptions()


logger = logging.getLogger(__name__)


def polygonize(
    input_path: Union[str, "os.PathLike[Any]"],
    output_path: Union[str, "os.PathLike[Any]"],
    output_layer: Optional[str] = None,
    value_column: str = "DN",
    dissolve: bool = True,
    simplify_tolerance: float = 0.0,
    simplify_algorithm: Union[SimplifyAlgorithm, str] = "rdp",
    simplify_lookahead: int = 8,
    simplify_preserve_common_boundaries: bool = False,
    dst_tiles_path: Union[str, "os.PathLike[Any]", None] = None,
    max_tile_size_mp: int = 500,
    nb_parallel: int = -1,
    force: bool = False,
):
    """Polygonize a raster file to a vector file.

    Based on the `max_tile_size_mb` parameter, the raster file will be processed tile by
    tile. If `dissolve_result` is True, the result will be dissolved based on the DN
    column.

    When a `simplify_tolerance` is specified, the result will be simplified. This will
    also be applied per tile to avoid huge temporary file storage requirements and to
    improve performance when `simplify_preserve_common_boundaries=True`.

    Args:
        input_path (PathLike): the input file.
        output_path (PathLike): the file to write the result to.
        output_layer (str, optional): output layer name. If None, the ``output_path``
            stem is used. Defaults to None.
        value_column (str, optional): the column name to use for the values in the
            output file. Defaults to "DN".
        dissolve (bool, optional): True to disolve the result, grouped on the
            `value_column`. Defaults to True.
        simplify_tolerance (float): tolerance to use for the simplification. Defaults to
            0.0, so no simplification. Depends on the ``algorithm`` specified.
            In projected coordinate systems this tolerance will typically be
            in meter, in geodetic systems this is typically in degrees.
        simplify_algorithm (str, optional): algorithm to use. Defaults to "rdp".

                * **"rdp"**: Ramer Douglas Peucker: tolerance is a distance
                * **"lang"**: Lang: tolerance is a distance
                * **"lang+"**: Lang, with extensions to further reduce number of points
                * **"vw"**: Visvalingam Whyatt: tolerance is an area

        simplify_lookahead (int, optional): used for Lang algorithms. Defaults to 8.
        simplify_preserve_common_boundaries (bool, optional): True to (try to) maintain
            common boundaries between all geometries in the input geometry list.
            Defaults to False.
        dst_tiles_path (PathLike, optional): the path to write the tiling scheme used
            to. Defaults to None, which avoids the tiles to be written.
        max_tile_size_mp (int, optional): to determine the number of tiles: the maximum
            number of pixels per tile in megapixels. Defaults to 500.
        nb_parallel (int, optional): the number of parallel processes to use.
            Defaults to -1: use all available CPUs.
        force (bool, optional): overwrite existing output file(s).
            Defaults to False.

    """
    # If output file exists already, either clean up or return...
    input_path = Path(input_path)
    output_path = Path(output_path)
    if _io_util.output_exists(path=output_path, remove_if_exists=force):
        return

    # Some inits
    start_time = datetime.now()
    if isinstance(simplify_algorithm, str):
        simplify_algorithm = SimplifyAlgorithm(simplify_algorithm)

    if output_layer is None:
        output_layer = fileops.get_default_layer(output_path)

    if nb_parallel == -1:
        nb_cpu = multiprocessing.cpu_count()
        nb_parallel = nb_cpu

    # Determine the bounding box of the input raster + the number of tiles to create
    input = gdal.OpenEx(input_path)
    xmin, xres, xskew, ymax, yskew, yres = input.GetGeoTransform()
    xmax = xmin + (input.RasterXSize * xres)
    ymin = ymax + (input.RasterYSize * yres)
    total_nb_pixels = input.RasterXSize * input.RasterYSize
    nb_squarish_tiles = total_nb_pixels // (max_tile_size_mp * 1024 * 1024)

    # Create temporary and output directory
    tmp_dir = _io_util.create_tempdir("geofileops/polygonize")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If the number of tiles is less than 2, no need to tile the raster
    if nb_squarish_tiles < 2:
        _polygonize_bbox(
            input_path=input_path,
            output_path=output_path,
            value_column=value_column,
            bbox=None,
            simplify_algorithm=simplify_algorithm,
            simplify_tolerance=simplify_tolerance,
            simplify_lookahead=simplify_lookahead,
            simplify_preserve_common_boundaries=simplify_preserve_common_boundaries,
            tmp_dir=tmp_dir,
        )
        return

    # Determine crs of the input file
    spatialref = input.GetSpatialRef()
    crs = pyproj.CRS(spatialref.ExportToWkt()) if spatialref is not None else None

    # Create a grid with the number of tiles as determined above
    tiles = gpd.GeoDataFrame(
        geometry=pygeoops.create_grid2((xmin, ymin, xmax, ymax), nb_squarish_tiles),
        crs=crs,
    )
    if dst_tiles_path is None:
        dst_tiles_path = tmp_dir / "tiles.gpkg"
    tiles.to_file(dst_tiles_path)

    # Polygonize + simplify each tile
    logger.info(f"Polygonize input file in {len(tiles)} tiles")

    try:
        with _processing_util.PooledExecutorFactory(
            threadpool=False,
            max_workers=nb_parallel,
            initializer=_processing_util.initialize_worker(),
        ) as calculate_pool:
            batches = {}
            polygonized_path = tmp_dir / output_path.name

            for i, tile in tiles.iterrows():
                tile_bbox = tile.geometry.bounds
                polygonize_part_path = tmp_dir / f"tile_{i}.gpkg"

                future = calculate_pool.submit(
                    _polygonize_bbox,
                    input_path=input_path,
                    output_path=polygonize_part_path,
                    value_column=value_column,
                    bbox=tile_bbox,
                    simplify_algorithm=simplify_algorithm,
                    simplify_tolerance=simplify_tolerance,
                    simplify_lookahead=simplify_lookahead,
                    simplify_preserve_common_boundaries=simplify_preserve_common_boundaries,
                    tmp_dir=tmp_dir,
                )
                batches[future] = polygonize_part_path

            # Loop till all parallel processes are ready, but process each one that is
            # ready already.
            # Remark: calculating can be done in parallel, but only one process can
            # write to the same output file at the time...
            start_time = datetime.now()
            nb_done = 0
            nb_batches = len(tiles)
            _general_util.report_progress(
                start_time,
                nb_done,
                nb_todo=nb_batches,
                operation="polygonize",
                nb_parallel=nb_parallel,
            )
            for future in futures.as_completed(batches):
                try:
                    message = future.result()
                    if message is not None:
                        logger.debug(message)

                    # If the calculate gave results, copy to output
                    tmp_partial_output_path = batches[future]
                    if (
                        tmp_partial_output_path.exists()
                        and tmp_partial_output_path.stat().st_size > 0
                    ):
                        fileops._append_to_nolock(
                            src=tmp_partial_output_path,
                            dst=polygonized_path,
                            dst_layer=output_layer,
                            create_spatial_index=False,
                        )
                        fileops.remove(tmp_partial_output_path)

                except Exception as ex:
                    message = f"Error {ex} executing {batches[future]}"
                    logger.exception(message)
                    raise RuntimeError(message) from ex

                # Log the progress and prediction speed
                nb_done += 1
                _general_util.report_progress(
                    start_time,
                    nb_done,
                    nb_todo=nb_batches,
                    operation="polygonize",
                    nb_parallel=nb_parallel,
                )

        # If no result file was created, stop further processing
        if not polygonized_path.exists():
            raise LeaveTry("No result file found")

        # Create spatial index if this is the default for the file
        if GeofileInfo(polygonized_path).default_spatial_index:
            fileops.create_spatial_index(
                path=polygonized_path,
                layer=output_layer,
                exist_ok=True,
                no_geom_ok=True,
            )

        # Dissolve the result if asked for
        if dissolve:
            geoops.dissolve(
                polygonized_path,
                output_path,
                explodecollections=True,
                groupby_columns=value_column,
            )
        else:
            fileops.move(polygonized_path, output_path)

    except LeaveTry:
        pass
    except Exception:
        fileops.remove(output_path, missing_ok=True)
        raise
    finally:
        if ConfigOptions.remove_temp_files:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Ready, took {datetime.now()-start_time}")


def _polygonize_bbox(
    input_path: Path,
    output_path: Path,
    value_column: str,
    bbox: Optional[tuple[float, float, float, float]],
    simplify_algorithm: SimplifyAlgorithm,
    simplify_tolerance: float,
    simplify_lookahead: int,
    simplify_preserve_common_boundaries: bool,
    tmp_dir: Path,
):
    # If a bbox is specified, first create a vrt file with that bbox
    if bbox is not None:
        vrt_path = tmp_dir / f"{output_path.stem}.vrt"
        options = gdal.TranslateOptions(
            projWin=[bbox[0], bbox[3], bbox[2], bbox[1]], format="VRT"
        )
        gdal.Translate(srcDS=str(input_path), destName=str(vrt_path), options=options)

        input_path = vrt_path

    if simplify_tolerance > 0.0:
        output_poly_path = tmp_dir / f"{output_path.stem}_poly.gpkg"
    else:
        output_poly_path = output_path

    # Polygonize
    start = time.perf_counter()
    gdal_polygonize.gdal_polygonize(
        src_filename=str(input_path),
        dst_filename=str(output_poly_path),
        dst_fieldname=value_column,
        quiet=True,
        layer_creation_options=["SPATIAL_INDEX=NO"],
    )
    logger.debug(f"Polygonize took {(time.perf_counter() - start):.2f} seconds")

    # Simplify if asked for
    if simplify_tolerance > 0.0:
        # If a bbox was specified, keep points on the bbox boundary
        keep_points_on = None
        if bbox is not None:
            keep_points_on = shapely.box(*bbox).boundary

        # Read, simplify, write
        start = time.perf_counter()
        poly_gdf = fileops.read_file(output_poly_path)
        poly_gdf.geometry = pygeoops.simplify(
            poly_gdf.geometry,
            algorithm=simplify_algorithm.value,
            tolerance=simplify_tolerance,
            lookahead=simplify_lookahead,
            preserve_common_boundaries=simplify_preserve_common_boundaries,
            keep_points_on=keep_points_on,
        )
        fileops.to_file(poly_gdf, output_path, append=True)
        logger.debug(f"Simplify took {(time.perf_counter() - start):.2f} seconds")

    # Cleanup tmp files
    if bbox is not None:
        try:
            vrt_path.unlink()
        except Exception:
            logger.debug(f"Could not remove temporary file: {vrt_path}")

    if simplify_tolerance > 0.0:
        try:
            fileops.remove(output_poly_path)
        except Exception:
            logger.debug(f"Could not remove temporary file: {output_poly_path}")
