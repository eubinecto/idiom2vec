from fsplit.filesplit import Filesplit
import argparse
import csv
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_splits_dir', type=str,
                        default="../../data/coca_spok/train_splits")
    parser.add_argument('--train_splits_fs_path', type=str,
                        default="../../data/coca_spok/train_splits/fs_manifest.csv")
    parser.add_argument('--train_ndjson_path', type=str,
                        default="../../data/coca_spok/train.ndjson")
    args = parser.parse_args()

    # --- build the manifest tsv --- #
    HEADER = "filename,filesize,encoding,header".split(",")
    with open(args.train_splits_fs_path, 'w') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(HEADER)  # write the header
        # get all the filenames of the splits.

        filenames = [
            name
            for name in os.listdir(args.train_splits_dir)
            if name.endswith('.ndjson')
        ]
        filenames = sorted(filenames,
                           key=lambda x: int(re.findall(r'_([0-9]+).ndjson', x)[0]),
                           reverse=False)
        file_sizes = [
            os.path.getsize(args.train_splits_dir + "/" + name)
            for name in filenames
        ]

        for name, file_size in zip(filenames, file_sizes):
            to_write = [name, file_size, "", ""]
            csv_writer.writerow(to_write)

    fs = Filesplit()
    fs.merge(input_dir=args.train_splits_dir, output_file=args.train_ndjson_path)


if __name__ == '__main__':
    main()
