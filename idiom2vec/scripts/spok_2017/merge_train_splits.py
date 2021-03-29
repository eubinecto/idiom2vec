"""
Merge the split files into a single .ndjson file.
"""
from fsplit.filesplit import Filesplit
from idiom2vec.configs import SPOK_2017_TRAIN_SPLITS_DIR, SPOK_2017_TRAIN_NDJSON


def main():
    fs = Filesplit()
    fs.merge(input_dir=SPOK_2017_TRAIN_SPLITS_DIR, output_file=SPOK_2017_TRAIN_NDJSON)


if __name__ == '__main__':
    main()
