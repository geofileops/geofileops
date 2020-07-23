
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops import geofile
from geofileops.util import geofileops_gpd

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_convexhull_gpkg(tmpdir):
    # Select some data from input to output file
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_convexhull(input_path, output_path)

def test_convexhull_shp(tmpdir):
    # Select some data from input to output file
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_convexhull(input_path, output_path)

def basetest_convexhull(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_gpd.convexhull(
            input_path=input_path,
            output_path=output_path)

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(output_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

def test_buffer_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_buffer(input_path, output_path)

def test_buffer_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_buffer(input_path, output_path)

def basetest_buffer(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_gpd.buffer(
            input_path=input_path,
            output_path=output_path,
            distance=1)

    # Now check if the output file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

def test_simplify_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_simplify(input_path, output_path)

def test_simplify_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_simplify(input_path, output_path)

def basetest_simplify(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_gpd.simplify(
            input_path=input_path,
            output_path=output_path,
            tolerance=5)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_select = geofile.getlayerinfo(input_path)
    assert layerinfo_orig.featurecount == layerinfo_select.featurecount
    assert len(layerinfo_orig.columns) == len(layerinfo_select.columns)

def test_dissolve_nogroupby_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_dissolve_nogroupby(input_path, output_path)

def test_dissolve_nogroupby_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_dissolve_nogroupby(input_path, output_path)

def basetest_dissolve_nogroupby(input_path, output_path):
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            explodecollections=True)

    # Now check if the result file is correctly created
    assert output_path.exists() == True
    layerinfo_orig = geofile.getlayerinfo(input_path)
    layerinfo_output = geofile.getlayerinfo(output_path)
    assert layerinfo_output.featurecount == 21
    assert len(layerinfo_output.columns) == 0

    # Now check the contents of the result file
    input_gdf = geofile.read_file(input_path)
    output_gdf = geofile.read_file(output_path)
    assert input_gdf.crs == output_gdf.crs
    assert len(output_gdf) == layerinfo_output.featurecount
    

def test_dissolve_groupby_gpkg(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    basetest_dissolve_groupby(input_path, output_path)

def test_dissolve_groupby_shp(tmpdir):
    # Buffer to test dir
    input_path = get_testdata_dir() / 'parcels.shp'
    output_path = Path(tmpdir) / 'parcels.shp'
    basetest_dissolve_groupby(input_path, output_path)

def basetest_dissolve_groupby(input_path, output_path):
    layerinfo_orig = geofile.getlayerinfo(input_path)
    geofileops_gpd.dissolve(
            input_path=input_path,
            output_path=output_path,
            groupby_columns=['GEWASGROEP'],
            explodecollections=False)

    # Now check if the tmp file is correctly created
    assert output_path.exists() == True
    layerinfo_output = geofile.getlayerinfo(output_path)
    assert layerinfo_output.featurecount == 6
    assert len(layerinfo_orig.columns) == len(layerinfo_output.columns)

if __name__ == '__main__':
    #Prepare tempdir
    import tempfile
    import shutil
    tmpdir = Path(tempfile.gettempdir()) / 'test_geofileops_gpd'
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    if not tmpdir.exists():
        tmpdir.mkdir()

    # Run
    test_dissolve_nogroupby_gpkg(tmpdir)
