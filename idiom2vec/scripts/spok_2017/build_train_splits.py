from typing import List
from identify_idioms.service import build_iip
from idiom2vec.configs import SPOK_2017_ORIGIN_SPLITS_DIR, SPOK_2017_TRAIN_SPLITS_DIR
import json
import re
import os
from multiprocessing import Pool

# the tokenizer to use.
iip = build_iip()


def remove_labels(line: str) -> str:
    """
    e.g. @@4171106
    """
    return re.sub(r'@@[0-9]{7} ', "", line).strip()


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


def process_split(split_origin_path: str):
    # path to train split
    split_train_name = split_origin_path.split("/")[-1].replace(".txt", ".ndjson")
    split_train_path = SPOK_2017_TRAIN_SPLITS_DIR.joinpath(split_train_name)
    # open one for reading in, one to write the processed file to
    with open(split_origin_path, 'r') as fh_r, open(split_train_path, 'w') as fh_w:
        for line in fh_r:
            cleansed = cleanse(line)
            tokens = process_line(cleansed)
            # write the tokens - newline delimited jsons
            fh_w.write(json.dumps(tokens) + "\n")

    print("Done:" + str(split_origin_path))


def main():
    # first, get all the txt splits
    split_origin_paths = [
        str(SPOK_2017_ORIGIN_SPLITS_DIR) + "/" + split_path
        for split_path in os.listdir(SPOK_2017_ORIGIN_SPLITS_DIR)
        if split_path.endswith(".txt")
    ]

    # multiprocessing.
    with Pool(4) as mp:
        mp.map(process_split, split_origin_paths)


if __name__ == '__main__':
    main()
