
import logging
import pandas as pd
import argparse
from config import DELIM


def count_ids(subs_ids_tsv_path: str, out_path: str):
    logger = logging.getLogger("exec")
    # first, read in the tsv file as a pandas dataframe
    logger.info("reading the sub ids file as dataframe")
    df = pd.read_csv(subs_ids_tsv_path, delimiter=DELIM)
    # group by the first column and count
    df = df.groupby(df.columns[0]).count()
    # write to the path
    df.to_csv(out_path, sep=DELIM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subs_ids_tsv_path", type=str,
                        help="path to the tsv file containing the ids of Opensub titles")
    parser.add_argument("out_path", type=str,
                        help="path to save the output file")
    args = parser.parse_args()
    count_ids(args.subs_ids_tsv_path, args.out_path)


if __name__ == '__main__':
    main()
