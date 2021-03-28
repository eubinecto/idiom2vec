from spacy import load
from spacy.tokens import Token

from config import NLP_MODEL
from idiom2vec.slide.utils import IdiomNLP, load_idiom_matcher
from termcolor import colored

# sentences to test
SENTENCES = (
    "What a lovely vase! You shouldn't have!",
    "Kate is willing to pay full price for an expensive handbag, but I just can't wrap my head around that.",
    "She wasn't that interested at first, but she loved it once she got the bit between her teeth.",
    # hyphenated
    "The bank's overdraft policy is a catch-22 for those trying to get out of poverty,"
    " as it charges you higher fees for having less money in your account.",
    "Established families tend to hold themselves above the johnny-come-lately.",
    # cased
    "Established families tend to hold themselves above the Johnny-come-lately.",
    # not hyphenated
    "The bank's overdraft policy is a catch 22 for those trying to get out of poverty,"
    " as it charges you higher fees for having less money in your account.",
    "Established families tend to hold themselves above the johnny come lately."
)


def main():
    nlp = load(NLP_MODEL)
    idiom_matcher = load_idiom_matcher()
    # this wrapper class
    idiom_nlp = IdiomNLP(nlp, idiom_matcher)

    for sent in SENTENCES:
        doc = idiom_nlp(sent)
        lemmas = [token.lemma_ for token in doc]
        poses = [token.pos_ for token in doc]
        lex_ids = [token.lex_id for token in doc]
        tags = [token.tag_ for token in doc]
        for lemma, poses, lex_ids, tag in zip(lemmas, poses, lex_ids, tags):
            if tag == "IDIOM":
                colored_lemma = colored(lemma, 'magenta')
            else:
                colored_lemma = colored(lemma, 'blue')
            print("({}, {}, {}, {}),".format(colored_lemma, poses, str(lex_ids), tag), end=" ")
        print("\n##########")


if __name__ == '__main__':
    main()
