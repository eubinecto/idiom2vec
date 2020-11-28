from spacy import load
from spacy.tokens import Token

from config import NLP_MODEL
from idiom2vec.slide.utils import IdiomNLP, load_idiom_matcher

# sentences to test
SENTENCES = (
    "What a lovely vase! You shouldn't have!",
    "Kate is willing to pay full price for an expensive handbag, but I just can't wrap my head around that.",
    "She wasn't that interested at first, but she loved it once she got the bit between her teeth."
)


def main():
    nlp = load(NLP_MODEL)
    idiom_matcher = load_idiom_matcher()
    # this wrapper class
    idiom_nlp = IdiomNLP(nlp, idiom_matcher)

    for sent in SENTENCES:
        doc = idiom_nlp(sent)
        lemmas = [token.lemma_ for token in doc]
        lex_ids = [token.lex_id for token in doc]
        print("original:", doc.text)
        print("lemmas:", lemmas)
        print("lex ids:", lex_ids)
        print("##########")


if __name__ == '__main__':
    main()
