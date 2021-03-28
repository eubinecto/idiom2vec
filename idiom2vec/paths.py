from pathlib import Path
LIB_DIR = Path(__file__).parent
ROOT_DIR = LIB_DIR.parent
DATA_DIR = ROOT_DIR.joinpath('data')
COCA_DIR = DATA_DIR.joinpath('coca')

# COCA_DIR files
SPOK_2017_TXT = COCA_DIR.joinpath('spok_2017.txt')  # this is the one to do prototyping on.
