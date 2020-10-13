
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from geofileops.util import ogr_util
from geofileops import geofile

class GdalBin():
    def __init__(self, set_gdal_bin: bool = True, gdal_bin_path: str = None):
        self.set_gdal_bin = set_gdal_bin
        if set_gdal_bin is True:
            if gdal_bin_path is None:
                self.gdal_bin_path = r"X:\GIS\Software\_Progs\OSGeo4W64_2020-05-29\bin"
            else:
                self.gdal_bin_path = gdal_bin_path

    def __enter__(self):
        if self.set_gdal_bin is True:
            import os
            os.environ['GDAL_BIN'] = self.gdal_bin_path

    def __exit__(self, type, value, traceback):
        #Exception handling here
        import os
        if os.environ['GDAL_BIN'] is not None:
            del os.environ['GDAL_BIN']

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def is_gdal_default_ok() -> bool:
    return False

def is_gdal_bin_ok() -> bool:
    return True

def test_get_gdal_to_use():

    # On windows, the default gdal installation with conda doesn't work
    if os.name == 'nt': 
        # If GDAL_BIN not set, should be not OK
        try:
            ogr_util.get_gdal_to_use('ST_area()')
            test_ok = True
        except:
            test_ok = False
        assert test_ok is is_gdal_default_ok(), "On windows, check is expected to be OK if GDAL_BIN is not set"

        # If GDAL_BIN set, it should be ok as well
        with GdalBin():
            try:
                ogr_util.get_gdal_to_use('ST_area()')
                test_ok = True
            except:
                test_ok = False
            assert test_ok is is_gdal_bin_ok(), "On windows, check is expected to be OK if GDAL_BIN is set (properly)"
        
    else:
        try:
            ogr_util.get_gdal_to_use('ST_area()')
            test_ok = True
        except:
            test_ok = False
        assert test_ok is True, "If not on windows, check is expected to be OK without setting GDAL_BIN"

if __name__ == '__main__':
    import tempfile
    tmpdir = tempfile.gettempdir()
    test_get_gdal_to_use()
