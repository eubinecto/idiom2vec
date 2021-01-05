# -- loads the raw subtitles, tokenize them using idiom nlp, save it as ndjson -- #
from config import SUBS_TOKENIZED_NDJSON_PATH, NLP_MODEL
from idiom2vec.opensub.utils import load_subs
import spacy
import json
from idiom2vec.slide.utils import IdiomNLP, load_idiom_matcher

nlp = spacy.load(NLP_MODEL)
idiom_matcher = load_idiom_matcher()
idiom_nlp = IdiomNLP(nlp=nlp, idiom_matcher=idiom_matcher)


def tokenize_sub(sub: str):
    global idiom_nlp
    doc = idiom_nlp(sub)
    # just get the lemmas
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop  # do I need this? should think about this later.
        if not token.is_punct  # don't need punctuations
        if not token.like_num  # don't need numbers
    ]
    return lemmas


def main():
    with open(SUBS_TOKENIZED_NDJSON_PATH, 'w') as fh:
        for idx, sub in enumerate(load_subs()):
            tokens = tokenize_sub(sub)
            to_write = json.dumps(tokens)
            fh.write(to_write + "\n")
            if idx > 0 and idx % 10000 == 0:
                print("done:" + str(idx + 1))


if __name__ == '__main__':
    main()
