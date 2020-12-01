# building
from idiom2vec.opensub.utils import load_sample
from termcolor import colored


def main():
    for example in load_sample():
        for context in example['contexts']:
            print(colored(context, 'magenta'))
        print("response:", colored(example['response'], 'blue'))
        print("######")


if __name__ == '__main__':
    main()
