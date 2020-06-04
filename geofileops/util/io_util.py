
import os
from pathlib import Path
import shutil
import tempfile

class CTError(Exception):
    def __init__(self, errors):
        self.errors = errors

def create_tempdir(base_dirname: str) -> Path:
    
    defaulttempdir = Path(tempfile.gettempdir())
    base_tempdir = defaulttempdir / base_dirname

    for i in range(1, 9999):
        try:
            tempdir = f"{str(base_tempdir)}_{i:04d}"
            os.mkdir(tempdir)
            return Path(tempdir)
        except FileExistsError:
            continue

    raise Exception(f"Wasn't able to create a temporary dir with basedir: {base_tempdir}")

def copyfile(src, dst):
    """
    standard shutil.copyfile is very slow on windows for large files.

    Args:
        src ([type]): [description]
        dst ([type]): [description]
    
    Raises:
        Exception: [description]
    """
    if os.name == 'nt':
        # On windows, this is a lot faster than all shutil alternatives
        #command = f'copy "{src}" "{dst}"'
        command = f'xcopy /j "{src}" "{dst}*"'
        returncode = os.system(command)
        if returncode != 0:
            raise Exception(f"Error executing {command}")
        
    else:
        buffer_size = 1024*1024*5
        with open(src, 'rb') as fsrc:
            with open(dst, 'wb') as fdest:
                shutil.copyfileobj(fsrc, fdest, buffer_size)
    
def copytree(src, dst, symlinks=False, ignore=[]):
    names = os.listdir(src)

    if not os.path.exists(dst):
        os.makedirs(dst)
    errors = []
    for name in names:
        if name in ignore:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore)
            else:
                copyfile(srcname, dstname)
            # XXX What about devices, sockets etc.?
        except (IOError, os.error) as ex:
            errors.append((srcname, dstname, str(ex)))
        except CTError as err:
            errors.extend(err.errors)
    if errors:
        raise CTError(errors)

def is_locked(filepath):
    """
    Checks if a file is locked by another process.
    """
    import msvcrt
    try:
        fd = os.open(filepath, os.O_APPEND | os.O_EXCL | os.O_RDWR)
    except OSError:
        return True

    try:
        filesize = os.path.getsize(filepath)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, filesize)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, filesize)
        os.close(fd)
        return False
    except (OSError, IOError):
        os.close(fd)
        return True
