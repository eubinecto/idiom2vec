

from pathlib import Path
from os import path
# the root directory of this project
# define the directories here
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(ROOT_DIR, "data")
# the dataset to use and test
C2DEF_CSV_PATH = path.join(DATA_DIR, "c2def.csv")
