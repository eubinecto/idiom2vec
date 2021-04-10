from fsplit.filesplit import Filesplit
from idiom2vec.configs import COCA_ORIGIN_TXT, COCA_ORIGIN_SPLITS_DIR, SPLIT_SIZE


def main():
    fs = Filesplit()
    fs.split(file=COCA_ORIGIN_TXT,
             split_size=SPLIT_SIZE,
             output_dir=COCA_ORIGIN_SPLITS_DIR,
             # this makes sure that there are no partial lines.
             newline=True)


if __name__ == '__main__':
    main()
