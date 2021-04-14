
from os import path
from pathlib import Path

# directories
HOME_DIR = path.expanduser("~")
DATA_DIR = path.join(HOME_DIR, 'data')
CORPORA_DIR = path.join(HOME_DIR, "corpora")  # this is always at home.
COCA_SPOK_DIR = path.join(CORPORA_DIR, "coca_spok")
COCA_MAG_DIR = path.join(CORPORA_DIR, "coca_mag")
COCA_FICT_DIR = path.join(CORPORA_DIR, "coca_fict")
OPENSUB_DIR = path.join(CORPORA_DIR, "opensub")
PROJECT_LIB_DIR = str(Path(__file__).resolve().parent)

# coca_spok
COCA_SPOK_CORPORA_DIR = path.join(COCA_SPOK_DIR, "corpora")
COCA_SPOK_ORIGIN_TXT_PATH = path.join(COCA_SPOK_DIR, 'origin.txt')
COCA_SPOK_ORIGIN_SPLITS_DIR = path.join(COCA_SPOK_DIR, 'origin_splits')
COCA_SPOK_TRAIN_SPLITS_DIR = path.join(COCA_SPOK_DIR, 'train_splits')
COCA_SPOK_TRAIN_SPLITS_FS_CSV_PATH = path.join(COCA_SPOK_TRAIN_SPLITS_DIR, 'fs_manifest.csv')
COCA_SPOK_TRAIN_NDJSON_PATH = path.join(COCA_SPOK_DIR, 'train.ndjson')

# coca_mag
COCA_MAG_CORPORA_DIR = path.join(COCA_MAG_DIR, "corpora")
COCA_MAG_ORIGIN_TXT_PATH = path.join(COCA_MAG_DIR, 'origin.txt')
COCA_MAG_ORIGIN_SPLITS_DIR = path.join(COCA_MAG_DIR, 'origin_splits')
COCA_MAG_TRAIN_SPLITS_DIR = path.join(COCA_MAG_DIR, 'train_splits')
COCA_MAG_TRAIN_SPLITS_FS_CSV_PATH = path.join(COCA_MAG_TRAIN_SPLITS_DIR, 'fs_manifest_csv')
COCA_MAG_TRAIN_NDJSON_PATH = path.join(COCA_MAG_DIR, 'train.ndjson')


# opensub
OPENSUB_CORPORA_DIR = path.join(OPENSUB_DIR, "corpora")
OPENSUB_ORIGIN_TXT_PATH = path.join(OPENSUB_DIR, 'origin.txt')
OPENSUB_ORIGIN_SPLITS_DIR = path.join(OPENSUB_DIR, 'origin_splits')
OPENSUB_TRAIN_SPLITS_DIR = path.join(OPENSUB_DIR, 'train_splits')
OPENSUB_TRAIN_SPLITS_FS_CSV_PATH = path.join(OPENSUB_TRAIN_SPLITS_DIR, 'fs_manifest_csv')
OPENSUB_TRAIN_NDJSON_PATH = path.join(OPENSUB_DIR, 'train.ndjson')

