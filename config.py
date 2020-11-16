

from pathlib import Path
from os import path
# the root directory of this project
# define the directories here
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(ROOT_DIR, "data")
WORD2VEC_DIR = path.join(DATA_DIR, "word2vec")
OPENSUB_DIR = path.join(DATA_DIR, "opensub")
PLAYGROUND_DIR = path.join(DATA_DIR, "playground")
GENSIM_PG_DIR = path.join(PLAYGROUND_DIR, "gensim_pg")
# the dataset to use and test
FASTTEXT_VEC_PATH = path.join(WORD2VEC_DIR, "crawl-300d-2M.vec")
FASTTEXT_PKL_PATH = path.join(WORD2VEC_DIR, "crawl-300d-2M.pkl")  # path to store the pickle binary


# delimiter to use for reading & writing data
DELIM = "\t"
