from typing import Union, Optional
import os.path

def find_root(root_indicator_file: str = ".ROOT",
              startpoint: Optional[str] = None):
    if startpoint is None:
        startpoint = os.getcwd()
    else:
        startpoint = os.path.normpath(os.path.abspath(startpoint))

    rootpath = check_dir_(root_indicator_file, startpoint)
    if rootpath is None:
        raise ValueError('ROOT directory not found until the root of the drive.')

    # return relative path in case of directory privacy
    return os.path.relpath(rootpath, os.curdir)

def check_dir_(root_filename: str, dirpath: str):
    if root_filename in os.listdir(dirpath):
        return dirpath

    # go upper level
    parent = get_upper_level_dir_(dirpath)
    if parent == dirpath:   # root of fs encountered
        return None

    return check_dir_(root_filename, parent)

def get_upper_level_dir_(dirpath: str) -> str:
    canonical_dirpath = os.path.normpath(os.path.abspath(dirpath))
    parent = os.path.join(canonical_dirpath, os.path.pardir)
    canonical_parent = os.path.normpath(parent)

    return canonical_parent




