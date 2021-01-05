from idiom2vec.opensub.utils import OpenSubCorpus
from idiom2vec.slide.utils import IdiomNLP, load_idiom_matcher
from config import NLP_MODEL, IDIOM2VEC_PKL_PATH
from spacy import load
import logging
import sys
import gensim
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    # prepare the corpus
    nlp = load(NLP_MODEL)
    idiom_matcher = load_idiom_matcher()
    idiom_nlp = IdiomNLP(nlp, idiom_matcher)
    opensub_corpus = OpenSubCorpus(idiom_nlp)
    # start training.. well you don't see any progress.. why?
    word2vec = gensim.models.Word2Vec(sentences=opensub_corpus, workers=10)
    print("training done")
    word2vec.save(IDIOM2VEC_PKL_PATH)


if __name__ == '__main__':
    main()
