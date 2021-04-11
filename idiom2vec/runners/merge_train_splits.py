from fsplit.filesplit import Filesplit
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_splits_dir', type=str)
    parser.add_argument('--train_ndjson_path', type=str)
    args = parser.parse_args()
    fs = Filesplit()
    fs.merge(input_dir=args.train_splits_dir, output_file=args.train_ndjson_path)


if __name__ == '__main__':
    main()