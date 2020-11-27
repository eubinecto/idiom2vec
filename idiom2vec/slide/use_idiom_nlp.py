from spacy import load
from config import NLP_MODEL
from idiom2vec.slide.utils.idiom_nlp import IdiomNLP


SENTENCES = (
    "What a lovely vase! You shouldn't have!",
    "Kate is willing to pay full price for an expensive handbag, but I just can't wrap my head around that."
)


def main():
    nlp = load(NLP_MODEL)
    idiom_nlp = IdiomNLP(nlp)

    for sent in SENTENCES:
        doc = idiom_nlp(sent)
        lemmas = [token.lemma_ for token in doc]
        print(lemmas)


if __name__ == '__main__':
    main()
