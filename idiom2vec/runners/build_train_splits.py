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
from joblib import Parallel, delayed
import argparse
import json
import os

# --- global vars --- #
iip = build_iip()  # identify-idioms pipeline
cleaner: Optional[Cleaner] = None  # domain-specific cleaner
train_splits_dir: Optional[str] = None  # to be used for progress tracking
origin_splits_dir: Optional[str] = None  # to be used for progress tracking
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
    global train_splits_dir
    split_origin_path = paths[0]
    split_train_path = paths[1]
    results = list()
    # --- process the results --- #
    with open(split_origin_path, 'r') as fh_r:
        for line in fh_r:
            processed = process_line_fn(line)
            results.append(json.dumps(processed))
    # --- write the results --- # 
    with open(split_train_path, 'w') as fh_w:
        # write the processed - newline delimited results
        for result in results:
            fh_w.write(result + "\n")
    # tracking progress here.
    print("Finished:" + str(split_origin_path))
    total = len(os.listdir(origin_splits_dir))
    done = len(os.listdir(train_splits_dir))
    print("Progress: {} / {}".format(done, total))


def load_read_paths():
    global origin_splits_dir, train_splits_dir
    todo_ids = set([
        todo_path.replace(".txt", "")
        for todo_path in os.listdir(origin_splits_dir)
        if todo_path.endswith(".txt")
    ])
    done_ids = set([
        done_path.replace(".ndjson", "")
        for done_path in os.listdir(train_splits_dir)
        if done_path.endswith(".ndjson")
    ])
    pending_ids = todo_ids - done_ids
    print("pending size:" + str(len(pending_ids)))
    return [
        os.path.join(origin_splits_dir, pending_id + ".txt")
        for pending_id in pending_ids
    ]
    

def main():
    global cleaner, origin_splits_dir, train_splits_dir, process_line_fn

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers',
                        type=int,
                        default=3)
    parser.add_argument('--corpus_name',
                        type=str,
                        default="coca_spok")
    args = parser.parse_args()
    # --- init the cleaner, process function and paths --- #
    if args.corpus_name == "coca_spok":
        print(args.corpus_name)
        cleaner = CocaSpokCleaner()
        process_line_fn = process_line_coca
        origin_splits_dir = COCA_SPOK_ORIGIN_SPLITS_DIR
        train_splits_dir = COCA_SPOK_TRAIN_SPLITS_DIR
    elif args.corpus_name == "coca_mag":
        cleaner = CocaMagCleaner()
        process_line_fn = process_line_coca
        origin_splits_dir = COCA_MAG_ORIGIN_SPLITS_DIR
        train_splits_dir = COCA_MAG_TRAIN_SPLITS_DIR
    elif args.corpus_name == "opensub":
        cleaner = OpenSubCleaner()
        process_line_fn = process_line_opensub
        origin_splits_dir = OPENSUB_ORIGIN_SPLITS_DIR
        train_splits_dir = OPENSUB_TRAIN_SPLITS_DIR
    else:
        raise ValueError("Invalid corpus name:" + args.corpus_name)

    # --- prepare the read & write paths --- #
    read_paths = load_read_paths()
    read_write_paths = [
        (read_path,
         train_splits_dir + "/" + read_path.split("/")[-1].replace(".txt", ".ndjson"))
        for read_path in read_paths
    ]

    # --- execute the process with parallelism --- #
    p = Parallel(n_jobs=args.num_workers)
    p(delayed(process_split)(paths) for paths in read_write_paths)


if __name__ == '__main__':
    main()
