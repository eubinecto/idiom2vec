"""
this is the main script, that will take ages, for sure.
"""
from typing import List, Tuple, Optional, Callable
from identify_idioms.service import build_iip
from idiom2vec.cleaners import Cleaner, CocaSpokCleaner, CocaMagCleaner, OpenSubCleaner
from idiom2vec.paths import (
    OPENSUB_ORIGIN_SPLITS_DIR,
    OPENSUB_TRAIN_SPLITS_DIR,
    COCA_SPOK_ORIGIN_SPLITS_DIR,
    COCA_SPOK_TRAIN_SPLITS_DIR,
    COCA_MAG_ORIGIN_SPLITS_DIR,
    COCA_MAG_TRAIN_SPLITS_DIR
)
from multiprocessing import Pool
import argparse
import json
import os

# --- global vars --- #
iip = build_iip()  # identify-idioms pipeline
cleaner: Optional[Cleaner] = None  # domain-specific cleaner
total: Optional[int] = None  # to be used for progress tracking
train_splits_dir: Optional[str] = None  # to be used for progress tracking
process_line_fn: Optional[Callable] = None


def process_line_coca(line: str) -> List[List[str]]:
    """
    split into sentences.
    Tokenise each sentence.
    Lemmatise and clean each token in the sentence.
    """
    global iip, cleaner
    cleaned = cleaner(line)
    sents = cleaned.split(".")
    # list of list of tokens. This is what we want.
    sents_processed = [
        [
            token.lemma_.replace(" ", "_") if token._.is_idiom else token.lemma_  # lemmatise them.
            for token in iip(sent)
            if len(token.text) > 1  # should be longer than 1
            if not token.is_stop  # don't need stop words
            if not token.is_punct  # don't need punctuations
            if not token.like_num  # don't need numbers
        ]
        for sent in sents
    ]
    return sents_processed


def process_line_opensub(line: str) -> List[str]:
    global iip, cleaner
    cleaned = cleaner(line)
    return [
        token.lemma_.replace(" ", "_") if token._.is_idiom else token.lemma_  # lemmatise them.
        for token in iip(cleaned)
        if len(token.text) > 1  # should be longer than 1
        if not token.is_stop  # don't need stop words
        if not token.is_punct  # don't need punctuations
        if not token.like_num  # don't need numbers
    ]


def process_split(paths: Tuple[str, str]):
    # path to train split
    global cleaner, total, train_splits_dir, process_line_fn
    split_origin_path = paths[0]
    split_train_path = paths[1]
    # open one for reading in, one to write the processed file to
    with open(split_origin_path, 'r') as fh_r, open(split_train_path, 'w') as fh_w:
        for line in fh_r:
            processed = process_line_fn(line)
            # write the processed - newline delimited jsons
            fh_w.write(json.dumps(processed) + "\n")

    # tracking progress here.
    print("Finished:" + str(split_origin_path))
    done = len(os.listdir(train_splits_dir))
    print("Progress: {} / {}".format(done, total))


def main():
    global cleaner, total, train_splits_dir, process_line_fn

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',
                        type=int)
    parser.add_argument('--corpus_name',
                        type=str)
    args = parser.parse_args()

    # --- prevent overwriting --- #
    if len(os.listdir(args.train_splits_dir)) > 1:
        raise ValueError("train_splits already exist")

    # --- init the global vars --- #
    total = len(os.listdir(args.origin_splits_dir))

    # --- init the cleaner & paths --- #
    if args.corpus_name == "coca_spok":
        cleaner = CocaSpokCleaner()
        process_line_fn = process_line_coca
        origin_splits_dir = COCA_SPOK_ORIGIN_SPLITS_DIR
        train_splits_dir = COCA_SPOK_TRAIN_SPLITS_DIR
    elif args.corpus_name == "coca_mag":
        cleaner = CocaMagCleaner()
        origin_splits_dir = COCA_MAG_ORIGIN_SPLITS_DIR
        train_splits_dir = COCA_MAG_TRAIN_SPLITS_DIR
        process_line_fn = process_line_coca
    elif args.corpus_name == "opensub":
        cleaner = OpenSubCleaner()
        origin_splits_dir = OPENSUB_ORIGIN_SPLITS_DIR
        train_splits_dir = OPENSUB_TRAIN_SPLITS_DIR
        process_line_fn = process_line_opensub
    else:
        raise ValueError("Invalid corpus name:" + args.corpus_name)

    # --- prepare the read & write paths --- #
    split_origin_paths = [
        origin_splits_dir + "/" + split_path
        for split_path in os.listdir(origin_splits_dir)
        if split_path.endswith(".txt")
    ]
    paths = [
        (split_origin_path,
         train_splits_dir + "/" + split_origin_path.split("/")[-1].replace(".txt", ".ndjson"))
        for split_origin_path in split_origin_paths
    ]

    # --- execute the process with multiprocessing --- #
    # https://stackoverflow.com/a/46843159
    p = Pool(args.num_workers)
    # run it asynchronously. This is more important.
    p.map_async(process_split, paths)
    p.close()
    p.join()


if __name__ == '__main__':
    main()
