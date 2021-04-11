from fsplit.filesplit import Filesplit
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_size',
                        type=int,
                        default=200000)
    parser.add_argument('--origin_txt_path',
                        type=str)
    parser.add_argument('--origin_splits_dir',
                        type=str)
    args = parser.parse_args()
    fs = Filesplit()
    fs.split(file=args.origin_txt_path,
             split_size=args.split_size,
             output_dir=args.origin_splits_dir,
             # this makes sure that there are no partial lines.
             newline=True)


if __name__ == '__main__':
    main()
