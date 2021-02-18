

# for global access
from spacy import load
from idiom2vec.config import NLP_MODEL
from idiom2vec.slide.utils import load_idiom_matcher, IdiomNLP
from idiom2vec.utils import load_resps
from termcolor import colored

IDIOM_MATCHER = load_idiom_matcher()
IDIOM_NLP = IdiomNLP(load(NLP_MODEL), IDIOM_MATCHER)


def main():
    global IDIOM_MATCHER
    tokens = (
        token
        for resp in load_resps()
        for token in IDIOM_NLP(resp)
        if not token.is_stop
        if not token.is_punct
        if not token.like_num
    )
    for idx, token in enumerate(tokens):
        if token.lemma in IDIOM_MATCHER._patterns:
            msg = "**{}**".format(colored(token, 'blue') + "|" + colored(token.lemma_, 'magenta'))
            print(msg, end=" ")
        else:
            print(token, end=" ")
        if idx % 20 == 0:
            print("\n", end="")


if __name__ == '__main__':
    main()

