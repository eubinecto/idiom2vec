# building
from idiom2vec.corpus.utils import load_sample_sub_ndjson
from termcolor import colored


def main():
    for example in load_sample_sub_ndjson():
        for context in example['contexts']:
            print(colored(context, 'magenta'))
        print(colored(example['response'], 'blue'))
        print("######")


if __name__ == '__main__':
    main()
