from spacy import load
from config import NLP_MODEL
from termcolor import colored

from idiom2vec.slide.utils import load_idiom_matcher, IdiomNLP

EXAMPLES = [
    "you need to know something",
    "They won't tell me anything else right now—they say all updates will be shared on a need-to-know basis.",
    "They won't tell me anything else right now—they say all updates will be shared on a need to know basis.",
]


def main():
    global EXAMPLES
    nlp = load(NLP_MODEL)
    idiom_matcher = load_idiom_matcher()
    idiom_nlp = IdiomNLP(nlp, idiom_matcher)
    for sent in EXAMPLES:
        print(sent)
        doc = idiom_nlp(sent)
        for token in doc:
            msg = "{}|{}".format(colored(token.text, 'blue'), colored(token.dep_, 'magenta'))
            print(msg, end=" ")
        else:
            print("\n", end="")
        print("#######")


if __name__ == '__main__':
    main()
