# building
from idiom2vec.utils import load_examples
from termcolor import colored


def main():
    for example in load_examples():
        for context in example['contexts']:
            print(colored(context, 'magenta'))
        print("response:", colored(example['response'], 'blue'))
        print("######")


if __name__ == '__main__':
    main()
