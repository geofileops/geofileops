# -*- coding: utf-8 -*-
"""
Module containing some utilities regarding io.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

class CTError(Exception):
    def __init__(self, errors):
        self.errors = errors

def create_tempdir(base_dirname: str) -> Path:
    
    defaulttempdir = Path(tempfile.gettempdir())

    for i in range(1, 999999):
        try:
            tempdir = defaulttempdir / f"{base_dirname}_{i:06d}"
            os.mkdir(tempdir)
            return Path(tempdir)
        except FileExistsError:
            continue

    raise Exception(f"Wasn't able to create a temporary dir with basedir: {defaulttempdir / base_dirname}") 

def get_tempfile_locked(
        base_filename: str,
        suffix: str = None,
        dirname: str = None,
        tempdir: Path = None) -> Tuple[Path, Path]:
    # If no dir specified, use default temp dir
    if tempdir is None:
        tempdir = Path(tempfile.gettempdir())
    if dirname is not None:
        tempdir = tempdir / dirname
        if not tempdir.exists():
            os.makedirs(tempdir)

    # Now look for a unique filename based on the base_filename and put a lock file
    for i in range(1, 999999):
        tempfile_path = tempdir / f"{base_filename}_{i:06d}{suffix}"
        tempfilelock_path = tempdir / f"{base_filename}_{i:06d}{suffix}.lock"
        result = create_file_atomic(tempfilelock_path)
        if result is True:
            if not tempfile_path.exists():
                # OK!
                return (tempfile_path, tempfilelock_path)
            else:
                # Apparently the lock file didn't exist, but the file did... so 
                # delete lock file and try again
                tempfilelock_path.unlink()

    raise Exception(f"Wasn't able to create a temporary file with base_filename: {base_filename}, dir: {dir}") 
 
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
        output = ''
        try:
            output = subprocess.check_output(command, shell=True)
        except Exception as ex:
            raise Exception(f"Error executing {command}, with output {output}") from ex
        
    else:
        buffer_size = 1024*1024*5
        with open(src, 'rb') as fsrc, \
             open(dst, 'wb') as fdest:
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

def create_file_atomic(filename) -> bool:
    """
    Create a lock file in an atomic way, so it is threadsafe.

    Returns True if the file was created by this thread, False if the file existed already.
    """
    try:
        fd = os.open(filename,  os.O_CREAT | os.O_EXCL)
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except IOError as ex:
        if ex.errno == 13:
            return False

    # If we get here, return False anyway       
    return False
