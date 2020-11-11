

from pathlib import Path
from os import path
# the root directory of this project
# define the directories here
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(ROOT_DIR, "data")
# the dataset to use and test
FASTTEXT_VEC_PATH = path.join(DATA_DIR, "crawl-300d-2M.vec")
FASTTEXT_PKL_PATH = path.join(DATA_DIR, "crawl-300d-2M.pkl")  # path to store the pickle binary
