import logging
from config import DELIM
import pandas as pd
from os import path
import argparse


def split_by_ids(cnt_tsv_path: str, subs_txt_path: str, out_path: str):
    # load this first
    cnt_df = pd.read_csv(cnt_tsv_path, delimiter=DELIM)
    # just select the second column
    with open(subs_txt_path, 'r') as r_fh:
        for _, row in cnt_df.iterrows():
            # this is the subtitle for a single movie
            sub_id = row[0].split(".")[0].replace("/", "_") + ".txt"
            cnt = row[1]
            lines = (r_fh.readline() for _ in range(cnt))
            # write the lines into a file
            with open(path.join(out_path, sub_id), 'w') as w_fh:
                w_fh.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cnt_tsv_path", type=str,
                        help="path to output of count_ids.py script")
    parser.add_argument("subs_txt_path", type=str,
                        help="path to the text file containing subtitles")
    parser.add_argument("out_path", type=str,
                        help="path to save the output file")
    args = parser.parse_args()
    split_by_ids(args.cnt_tsv_path, args.subs_txt_path, args.out_path)


if __name__ == '__main__':
    main()
