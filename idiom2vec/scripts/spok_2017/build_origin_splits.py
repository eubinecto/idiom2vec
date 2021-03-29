from fsplit.filesplit import Filesplit
from idiom2vec.configs import SPOK_2017_ORIGIN_TXT, SPOK_2017_ORIGIN_SPLITS_DIR, SPLIT_SIZE


def main():
    fs = Filesplit()
    fs.split(file=SPOK_2017_ORIGIN_TXT,
             split_size=SPLIT_SIZE,
             output_dir=SPOK_2017_ORIGIN_SPLITS_DIR,
             # this makes sure that there are no partial lines.
             newline=True)


if __name__ == '__main__':
    main()
