# take care before importing this module, it will modify the project PYTHONPATH and may break the dependencies

import logging
import sys
import os.path as osp
from .root_finder import find_root

try:
    project_src_path = find_root('.SRC')
    project_src_path = osp.abspath(project_src_path)
    if project_src_path not in sys.path:
        sys.path.insert(0, project_src_path)
        logging.warning(f"Prepend to sys.path the directory {project_src_path} because of manual imports. ")
    else:
        logging.debug(f"sys.path has already been containing {project_src_path}")

except ValueError:
    logging.warning(f"You'd called to prepend the directory containing the file named as '.SRC' to PYTHONPATH. "
                    f"But the directory is not found. This import will thus do nothing.")
