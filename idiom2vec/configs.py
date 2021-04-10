from pathlib import Path

# The directories
LIB_DIR = Path(__file__).parent
ROOT_DIR = LIB_DIR.parent
DATA_DIR = ROOT_DIR.joinpath('data')
COCA_DIR = DATA_DIR.joinpath('coca')
COCA_ORIGIN_SPLITS_DIR = COCA_DIR.joinpath('origin_splits')
COCA_TRAIN_SPLITS_DIR = COCA_DIR.joinpath('train_splits')
COCA_CORPORA_DIR = COCA_DIR.joinpath('corpora')
IDIOM2VEC_DIR = DATA_DIR.joinpath("idiom2vec")

# COCA_DIR - files
COCA_ORIGIN_TXT = COCA_DIR.joinpath('origin.txt')  # this is the one to do prototyping on.
COCA_TRAIN_NDJSON = COCA_DIR.joinpath('train.ndjson')  # the processed tokens to be used for training.

# COCA_TRAIN_SPLITS files
COCA_TRAIN_SPLITS_FS_MANIFEST_CSV = COCA_TRAIN_SPLITS_DIR.joinpath('fs_manifest.csv')

# configs for file split
SPLIT_SIZE = 200000

# Idiom2vec files
IDIOM2VEC_001_MODEL = IDIOM2VEC_DIR.joinpath('idiom2vec_001.model')  # 0.0.1  (20 epochs)
IDIOM2VEC_002_MODEL = IDIOM2VEC_DIR.joinpath('idiom2vec_002.model')  # 0.0.2  (increasing the corpus size)
