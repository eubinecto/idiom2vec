# use multiprocessing to speed up the parsing process
import multiprocessing as mp
import logging
import json
import argparse
import os
from sys import stdout
logging.basicConfig(stream=stdout, level=logging.INFO)
# to get the names of all files under the splits directory
NUM_PROC = 5
MIN_LENGTH = 2
NUM_CONTEXTS = 10


def setup_logger(logger):
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)


def should_skip(line) -> bool:
    global MIN_LENGTH
    if not line:
        return True
    if len(line) < MIN_LENGTH:
        return True
    # else, return false
    return False


def preproc_line(line) -> str:
    # simple preprocessing.
    return line.strip()


def to_ndjson(zipped_path):
    """
    parses a batch (all subtitles for a movie) to ndjson,
    and write ndjson to the output path.
    """
    logger = logging.getLogger("to_ndjson")
    setup_logger(logger)
    # unpack the zipped path
    batch_path, out_path = zipped_path
    logger = logging.getLogger("to_ndjson")
    # init a bucket for previous lines
    prev_lines = list()
    # open the two files
    with open(batch_path, 'r') as r_fh, open(out_path, 'w') as w_fh:
        for line in r_fh:
            if should_skip(line):
                logger.warning("SKIP:{}".format(line))
                continue
            response = preproc_line(line)
            example = {
                "response": response,
                "contexts": prev_lines
            }
            # ndjson
            to_write = [json.dumps(example), "\n"]
            # write lines
            w_fh.writelines(to_write)
            # prepare prev lines, for the next response
            prev_lines.append(response)
            if len(prev_lines) > NUM_CONTEXTS + 1:
                # pop the first prev line
                prev_lines.pop(0)
        else:
            logger.info("DONE:batch_path={}|out_path={}".format(batch_path, out_path))


def main():
    global NUM_PROC
    parser = argparse.ArgumentParser()
    parser.add_argument("splits_path", type=str,
                        help="path to the directory with all ")
    parser.add_argument("splits_ndjson_path", type=str,
                        help="path to save the output file")
    # list of paths to all batches
    args = parser.parse_args()
    batch_paths = [
        os.path.join(args.splits_path, txt_file)
        for txt_file in os.listdir(args.splits_path)
    ]
    # create output paths
    out_paths = [
        os.path.join(args.splits_ndjson_path, batch_path.split("/")[-1].replace(".txt", ".ndjson"))
        for batch_path in batch_paths
    ]
    zipped_paths = zip(batch_paths, out_paths)

    with mp.Pool(processes=NUM_PROC) as p:
        p.map(to_ndjson, zipped_paths)


if __name__ == '__main__':
    main()
