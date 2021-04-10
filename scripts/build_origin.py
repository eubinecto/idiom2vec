"""
merge all the corpus into one text.
"""
import glob
from idiom2vec.configs import COCA_CORPORA_DIR, COCA_ORIGIN_TXT


def main():
    files = glob.glob(str(COCA_CORPORA_DIR) + "/*.txt", recursive=True)

    with open(COCA_ORIGIN_TXT, "w") as outfile:
        for f in files:
            with open(f, "r") as infile:
                outfile.write(infile.read())


if __name__ == '__main__':
    main()
