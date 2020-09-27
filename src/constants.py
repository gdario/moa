from pathlib import Path
import platform

if platform.system() == "Darwin":
    BASE_DIR = '/Users/dariog/Projects/moa'
else:
    BASE_DIR = '/home/giovenko/Projects/moa'

BASE_DIR = Path(BASE_DIR)
DATA_DIR = BASE_DIR/'data'
