from fsplit.filesplit import Filesplit
import argparse
import os

from idiom2vec.paths import (
    COCA_SPOK_ORIGIN_SPLITS_DIR,
    COCA_SPOK_ORIGIN_TXT_PATH,
    COCA_MAG_ORIGIN_SPLITS_DIR,
    COCA_MAG_ORIGIN_TXT_PATH,
    OPENSUB_ORIGIN_SPLITS_DIR,
    OPENSUB_ORIGIN_TXT_PATH
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_size',
                        type=int,
                        default=200000)
    parser.add_argument('--corpus_name',
                        type=str)
    args = parser.parse_args()

    # --- init the paths --- #
    if args.corpus_name == "coca_spok":
        origin_splits_dir = COCA_SPOK_ORIGIN_SPLITS_DIR
        origin_txt_path = COCA_SPOK_ORIGIN_TXT_PATH
    elif args.corpus_name == "coca_mag":
        origin_splits_dir = COCA_MAG_ORIGIN_SPLITS_DIR
        origin_txt_path = COCA_MAG_ORIGIN_TXT_PATH
    elif args.corpus_name == "opensub":
        origin_splits_dir = OPENSUB_ORIGIN_SPLITS_DIR
        origin_txt_path = OPENSUB_ORIGIN_TXT_PATH
    else:
        raise ValueError("Invalid corpus name:" + args.corpus_name)

    fs = Filesplit()
    fs.split(file=origin_txt_path,
             split_size=args.split_size,
             output_dir=origin_splits_dir,
             # this makes sure that there are no partial lines.
             newline=True)


if __name__ == '__main__':
    main()
