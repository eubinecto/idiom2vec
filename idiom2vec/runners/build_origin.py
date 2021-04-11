"""
merge all the corpus into one text.
"""
import glob
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpora_dir', type=str)
    parser.add_argument('--origin_txt_path', type=str)
    args = parser.parse_args()
    # get all the files
    files = glob.glob(str(args.corpora_dir) + "/*.txt", recursive=True)
    with open(args.corpora_dir, "w") as outfile:
        for f in files:
            with open(f, "r") as infile:
                outfile.write(infile.read())


if __name__ == '__main__':
    main()
