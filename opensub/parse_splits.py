# use multiprocessing to speed up the parsing process
import multiprocessing as mp
import logging
import json
import argparse
# to get the names of all files under the splits directory
from os import walk


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("splits_path", type=str,
                        help="path to the directory with all ")
    parser.add_argument("out_path", type=str,
                        help="path to save the output file")


if __name__ == '__main__':
    main()
