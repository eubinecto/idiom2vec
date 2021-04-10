"""
this is the main script, that will take ages, for sure.
"""
import csv
from typing import List, Tuple
from identify_idioms.service import build_iip
import json
import re
import os
from multiprocessing import Pool
import argparse


# the tokenizer to use.
iip = build_iip()

HEADER = "filename,filesize,encoding,header".split(",")


def remove_labels(line: str) -> str:
    """
    e.g. @@4171106
    """
    return re.sub(r'[0-9]{7} ', "", line).strip()


def replace_profanities(line: str) -> str:
    """
    e.g. they will add @ @ @ @ @ @ @ @ @ @ over the next
    """
    return line.replace('@ @ @ @ @ @ @ @ @ @', 'PROFANITY').strip()


def remove_nonverbals(line: str) -> str:
    """
    e.g. 10 minutes ( ph ) have
    """
    return re.sub(r'\(.+?\)', "", line).strip()


def remove_speakers(line: str) -> str:
    """
    e.g. @!DAVID-GREENE#
    e.g. -MTP-DA#
    """
    line = re.sub(r'@![a-zA-Z-]+# |@![a-zA-Z-]+ ', "", line).strip()
    line = re.sub(r'[A-Z-]+#', "", line).strip()
    return line


def cleanse(line: str) -> str:
    line = remove_labels(line)
    line = replace_profanities(line)
    line = remove_speakers(line)
    line = remove_nonverbals(line)
    return line


def process_line(line: str) -> List[str]:
    """
    tokenise, lemmatise, filter out.
    """
    global iip
    processed = [
        token.lemma_  # lemmatise
        for token in iip(line)  # tokenise
        if len(token.text) > 1  # should be longer than 1
        if not token.is_stop  # don't need stop words
        if not token.is_punct  # don't need punctuations
        if not token.like_num  # don't need numbers
    ]
    return processed


def process_split(paths: Tuple[str, str]):
    # path to train split
    split_origin_path = paths[0]
    split_train_path = paths[1]
    # open one for reading in, one to write the processed file to
    with open(split_origin_path, 'r') as fh_r, open(split_train_path, 'w') as fh_w:
        for line in fh_r:
            cleansed = cleanse(line)
            tokens = process_line(cleansed)
            # write the tokens - newline delimited jsons
            fh_w.write(json.dumps(tokens) + "\n")

    print("Done:" + str(split_origin_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',
                        type=int,
                        default=12)
    parser.add_argument('--coca_origin_splits_dir',
                        type=str,
                        default="../data/coca/origin_splits")
    parser.add_argument('--coca_train_splits_dir',
                        type=str,
                        default="../data/coca/train_splits")
    parser.add_argument('--coca_train_splits_fs',
                        type=str,
                        default="../data/coca/train_splits/fs_manifest.csv")
    args = parser.parse_args()

    global HEADER
    # first, get all the txt splits
    split_origin_paths = [
        args.coca_origin_splits_dir + "/" + split_path
        for split_path in os.listdir(args.coca_origin_splits_dir)
        if split_path.endswith(".txt")
    ]
    paths = [
        args.coca_train_splits_dir + "/" + split_origin_path.split("/")[-1].replace(".txt", ".ndjson")
        for split_origin_path in split_origin_paths
    ]

    # multiprocessing.
    with Pool(args.num_workers) as mp:
        mp.map(process_split, paths)

    # then build the train_split manifests
    with open(args.coca_train_splits_fs, 'w') as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(HEADER)  # write the header
        # get all the filenames of the splits.

        filenames = [
            name
            for name in os.listdir(args.coca_train_splits_dir)
            if name.endswith('.ndjson')
        ]
        filenames = sorted(filenames,
                           key=lambda x: int(re.findall(r'_([0-9]+).ndjson', x)[0]),
                           reverse=False)
        file_sizes = [
            os.path.getsize(args.coca_train_splits_dir + "/" + name)
            for name in filenames
        ]

        for name, file_size in zip(filenames, file_sizes):
            to_write = [name, file_size, "", ""]
            csv_writer.writerow(to_write)


if __name__ == '__main__':
    main()
