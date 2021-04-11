"""
this is the main script, that will take ages, for sure.
"""
from typing import List, Tuple, Optional
from identify_idioms.service import build_iip
from idiom2vec.cleaners import Cleaner, CocaSpokCleaner, CocaMagCleaner
from multiprocessing import Pool
import argparse
import json
import csv
import os
import re

# --- global vars --- #
iip = build_iip()
cleaner: Optional[Cleaner] = None  # to be filled


def process_line(line: str) -> List[str]:
    """
    tokenise, lemmatise, filter out.
    """
    global iip, cleaner
    cleaned = cleaner(line)
    processed = [
        token.lemma_  # lemmatise
        for token in iip(cleaned)  # tokenise
        if len(token.text) > 1  # should be longer than 1
        if not token.is_stop  # don't need stop words
        if not token.is_punct  # don't need punctuations
        if not token.like_num  # don't need numbers
    ]
    return processed


def process_split(paths: Tuple[str, str]):
    # path to train split
    global cleaner
    split_origin_path = paths[0]
    split_train_path = paths[1]
    # open one for reading in, one to write the processed file to
    with open(split_origin_path, 'r') as fh_r, open(split_train_path, 'w') as fh_w:
        for line in fh_r:
            tokens = process_line(line)
            # write the tokens - newline delimited jsons
            fh_w.write(json.dumps(tokens) + "\n")
    print("Done:" + str(split_origin_path))


def main():
    global cleaner
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',
                        type=int)
    parser.add_argument('--corpus_name',
                        type=str)
    parser.add_argument('--origin_splits_dir',
                        type=str)
    parser.add_argument('--train_splits_dir',
                        type=str)
    parser.add_argument('--train_splits_fs_path',
                        type=str)
    args = parser.parse_args()

    # --- prevent overwriting --- #
    if len(os.listdir(args.train_splits_dir)) > 1:
        raise ValueError("train_splits already exist")

    # --- init the cleaner --- #
    if args.corpus_name == "coca_spok":
        cleaner = CocaSpokCleaner()
    elif args.corpus_name == "coca_mag":
        cleaner = CocaMagCleaner()
    else:
        raise ValueError("Invalid corpus name:" + args.corpus_name)

    # --- prepare the read & write paths --- #
    split_origin_paths = [
        args.origin_splits_dir + "/" + split_path
        for split_path in os.listdir(args.origin_splits_dir)
        if split_path.endswith(".txt")
    ]
    paths = [
        (split_origin_path,
         args.train_splits_dir + "/" + split_origin_path.split("/")[-1].replace(".txt", ".ndjson"))
        for split_origin_path in split_origin_paths
    ]

    # --- execute the process with multiprocessing --- #
    with Pool(args.num_workers) as mp:
        mp.map(process_split, paths)

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


if __name__ == '__main__':
    main()
