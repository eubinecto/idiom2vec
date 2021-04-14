from fsplit.filesplit import Filesplit
from idiom2vec.paths import (
    COCA_SPOK_TRAIN_SPLITS_DIR,
    COCA_SPOK_TRAIN_NDJSON_PATH,
    COCA_SPOK_TRAIN_SPLITS_FS_CSV_PATH,
    COCA_MAG_TRAIN_SPLITS_DIR,
    COCA_MAG_TRAIN_NDJSON_PATH,
    COCA_MAG_TRAIN_SPLITS_FS_CSV_PATH,
    OPENSUB_TRAIN_SPLITS_DIR,
    OPENSUB_TRAIN_NDJSON_PATH,
    OPENSUB_TRAIN_SPLITS_FS_CSV_PATH
)
import argparse
import csv
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_name', type=str,
                        default="opensub")
    args = parser.parse_args()
    
    # --- init the cleaner & paths --- #
    if args.corpus_name == "coca_spok":
        train_splits_dir = COCA_SPOK_TRAIN_SPLITS_DIR
        train_splits_fs_path = COCA_SPOK_TRAIN_SPLITS_FS_CSV_PATH
        train_ndjson_path = COCA_SPOK_TRAIN_NDJSON_PATH
    elif args.corpus_name == "coca_mag":
        train_splits_dir = COCA_MAG_TRAIN_SPLITS_DIR
        train_splits_fs_path = COCA_MAG_TRAIN_SPLITS_FS_CSV_PATH
        train_ndjson_path = COCA_MAG_TRAIN_NDJSON_PATH
    elif args.corpus_name == "opensub":
        train_splits_dir = OPENSUB_TRAIN_SPLITS_DIR
        train_splits_fs_path = OPENSUB_TRAIN_SPLITS_FS_CSV_PATH
        train_ndjson_path = OPENSUB_TRAIN_NDJSON_PATH
    else:
        raise ValueError("Invalid corpus name:" + args.corpus_name)

    # --- build the manifest tsv --- #
    HEADER = "filename,filesize,encoding,header".split(",")
    with open(train_splits_fs_path, 'w') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(HEADER)  # write the header
        # get all the filenames of the splits.

        filenames = [
            name
            for name in os.listdir(train_splits_dir)
            if name.endswith('.ndjson')
        ]
        filenames = sorted(filenames,
                           key=lambda x: int(re.findall(r'_([0-9]+).ndjson', x)[0]),
                           reverse=False)
        file_sizes = [
            os.path.getsize(train_splits_dir + "/" + name)
            for name in filenames
        ]

        for name, file_size in zip(filenames, file_sizes):
            to_write = [name, file_size, "", ""]
            csv_writer.writerow(to_write)

    # then merge them into a single file.
    fs = Filesplit()
    fs.merge(input_dir=train_splits_dir, output_file=train_ndjson_path)


if __name__ == '__main__':
    main()
