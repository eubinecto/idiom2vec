import csv
import os
import re
from idiom2vec.configs import SPOK_2017_TRAIN_SPLITS_FS_MANIFEST_CSV, SPOK_2017_TRAIN_SPLITS_DIR, SPLIT_SIZE

HEADER = "filename,filesize,encoding,header".split(",")


def main():
    global HEADER

    with open(SPOK_2017_TRAIN_SPLITS_FS_MANIFEST_CSV, 'w') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(HEADER)  # write the header
        # get all the filenames of the splits.
        filenames = [
            name
            for name in os.listdir(SPOK_2017_TRAIN_SPLITS_DIR)
            if name.endswith('.ndjson')
        ]
        filenames = sorted(filenames,
                           key=lambda x: int(re.findall(r'_([0-9]+).ndjson', x)[0]),
                           reverse=False)

        for name in filenames:
            to_write = [name, SPLIT_SIZE, "", ""]
            csv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
