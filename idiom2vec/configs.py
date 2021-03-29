from pathlib import Path

# The directories
LIB_DIR = Path(__file__).parent
ROOT_DIR = LIB_DIR.parent
DATA_DIR = ROOT_DIR.joinpath('data')
SPOK_2017_DIR = DATA_DIR.joinpath('spok_2017')
SPOK_2017_ORIGIN_SPLITS_DIR = SPOK_2017_DIR.joinpath('origin_splits')
SPOK_2017_TRAIN_SPLITS_DIR = SPOK_2017_DIR.joinpath('train_splits')
IDIOM2VEC_DIR = DATA_DIR.joinpath("idiom2vec")

# SPOK_2017 files
SPOK_2017_ORIGIN_TXT = SPOK_2017_DIR.joinpath('origin.txt')  # this is the one to do prototyping on.
SPOK_2017_TRAIN_NDJSON = SPOK_2017_DIR.joinpath('train.ndjson')  # the processed tokens to be used for training.

# SPOK_2017_TRAIN_SPLITS files
SPOK_2017_TRAIN_SPLITS_FS_MANIFEST_CSV = SPOK_2017_TRAIN_SPLITS_DIR.joinpath('fs_manifest.csv')

# configs for file split
SPLIT_SIZE = 200000
