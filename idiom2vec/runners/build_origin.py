"""
merge all the corpus into one text.
"""
import glob
import argparse

from idiom2vec.paths import (
    COCA_SPOK_ORIGIN_TXT_PATH, COCA_MAG_ORIGIN_TXT_PATH, OPENSUB_ORIGIN_TXT_PATH,
    COCA_SPOK_CORPORA_DIR, COCA_MAG_CORPORA_DIR, OPENSUB_CORPORA_DIR
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpora_dir', type=str)
    parser.add_argument('--origin_txt_path', type=str)
    parser.add_argument('--corpus_name', type=str)
    args = parser.parse_args()
    # --- init the paths --- #
    if args.corpus_name == "coca_spok":
        origin_txt_path = COCA_SPOK_ORIGIN_TXT_PATH
        corpora_dir = COCA_SPOK_CORPORA_DIR
    elif args.corpus_name == "coca_mag":
        origin_txt_path = COCA_MAG_ORIGIN_TXT_PATH
        corpora_dir = COCA_MAG_CORPORA_DIR
    elif args.corpus_name == "opensub":
        origin_txt_path = OPENSUB_ORIGIN_TXT_PATH
        corpora_dir = OPENSUB_CORPORA_DIR
    else:
        raise ValueError("Invalid corpus name:" + args.corpus_name)

    # get all the files
    files = glob.glob(str(corpora_dir) + "/*.txt", recursive=True)
    with open(origin_txt_path, "w") as outfile:
        for f in files:
            with open(f, "r") as infile:
                outfile.write(infile.read())


if __name__ == '__main__':
    main()
