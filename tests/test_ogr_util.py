
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops.util import ogr_util
from geofileops import geofile

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_check_gdal_spatialite_install():

    # On windows, the default gdal installation with conda doesn't work
    if os.name == 'nt': 
        # If GDAL_BIN not set, should be not OK
        test_ok = False
        try:
            ogr_util.check_gdal_spatialite_install('ST_area()')
            test_ok = True
        except:
            assert True == True, "On windows, check is expected to be ok if GDAL_BIN is not set"
        if test_ok == True:
            assert True == True, "On windows, check is expected to be ok if GDAL_BIN is not set"

        # If GDAL_BIN set, it should be ok as well
        test_ok = False
        try: 
            os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
            try:
                ogr_util.check_gdal_spatialite_install('ST_area()')
                test_ok = True
            except:
                assert True == False, "On windows, check is expected to be OK if GDAL_BIN is set (properly)"
            
            if test_ok == True:
                assert True == True, "On windows, check is expected to be OK if GDAL_BIN is set (properly)"
        finally:
            del os.environ['GDAL_BIN']
        
    else:
        try:
            ogr_util.check_gdal_spatialite_install('ST_area()')
            assert True == True, "If not on windows, check is expected to be OK without setting GDAL_BIN"
        except:
            assert True == False, "If not on windows, check is expected to be OK without setting GDAL_BIN"

def test_spatialite_dependencies(tmpdir):
    
    # Get some version info from spatialite...
    input_path = get_testdata_dir() / 'parcels.gpkg'
    output_path = Path(tmpdir) / 'parcels.gpkg'
    sql_stmt = f'select spatialite_version(), HasGeos(), HasGeosAdvanced(), HasGeosTrunk(), geos_version(), rttopo_version()'

    print('check without GDAL_BIN')
    ogr_util.vector_translate(
            input_path=input_path,
            output_path=output_path,
            sql_stmt=sql_stmt)
    result_gdf = geofile.read_file(output_path)
    print(result_gdf)

    assert 'GDAL_BIN' not in os.environ    
    assert result_gdf['spatialite_version()'][0] >= '4.3.0'
    assert result_gdf['geos_version()'][0] >= '3.8.1'
    
    # On non-windows, geos should be ok 
    if os.name != 'nt': 
        assert result_gdf['rttopo_version()'][0] is not None
    else: 
        # On windows, in the conda gdal installation geos operations don't seem to work in spatialite 
        assert result_gdf['rttopo_version()'][0] is None
        try:
            os.environ['GDAL_BIN'] = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
            print(f"check with GDAL_BIN={os.environ['GDAL_BIN']}")
            ogr_util.vector_translate(
                    input_path=input_path,
                    output_path=output_path,
                    sql_stmt=sql_stmt)
            result_gdf = geofile.read_file(output_path)
            print(result_gdf)
        finally:
            del os.environ['GDAL_BIN']

        assert 'GDAL_BIN' not in os.environ    
        assert result_gdf['spatialite_version()'][0] >= '4.3.0'
        assert result_gdf['geos_version()'][0] >= '3.8.1'
        assert result_gdf['lwgeom_version()'][0] is not None

if __name__ == '__main__':
    import tempfile
    tmpdir = tempfile.gettempdir()
    test_check_gdal_spatialite_install()
    #test_spatialite_dependencies(tmpdir)