import logging
import pickle

from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors

from idiom2vec.config import FASTTEXT_PKL_PATH, FASTTEXT_VEC_PATH
from os import path
import sys


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def load_fasttext_model() -> Word2VecKeyedVectors:
    """
    load the model from cache if stored already. if not, build the model. (will take about 10 minutes)
    :return:
    """
    logger = logging.getLogger("load_fasttext_model")
    if path.exists(FASTTEXT_PKL_PATH):
        with open(FASTTEXT_PKL_PATH, 'rb') as fh:
            logger.info("loading from cache...:{}".format(FASTTEXT_PKL_PATH))
            fasttext_model = pickle.loads(fh.read())
            return fasttext_model
    else:
        fasttext_model = KeyedVectors.load_word2vec_format(fname=FASTTEXT_VEC_PATH, binary=False)
        # cache it to use it later
        with open(FASTTEXT_PKL_PATH, 'wb') as fh:
            fh.write(pickle.dumps(fasttext_model))
        # then return it
        return fasttext_model


FASTTEXT_MODEL = load_fasttext_model()


def analogize(A: str, B: str, C: str):
    """
    this will return X, where:
    A - B = C - X. i.e. vector_offset(A, B) = vector_offset(C, X).
    that is, X = B - A + C
    """
    global FASTTEXT_MODEL
    A_vec = FASTTEXT_MODEL.get_vector(A)
    B_vec = FASTTEXT_MODEL.get_vector(B)
    C_vec = FASTTEXT_MODEL.get_vector(C)
    # why does the "ratio equation" not work?
    # A: Because they are vectors! and what you need is vector arithmetics.
    x = B_vec - A_vec + C_vec
    for sim in FASTTEXT_MODEL.similar_by_vector(x, topn=20):
        print(sim)
    print("########")


def main():
    global FASTTEXT_MODEL
    analogize("Korea", "Seoul", "UK")
    analogize("Korea", "Seoul", "Apple")  # what if A and C are not of the same entity?
    analogize("Apple", "iPhone", "Samsung")
    analogize("chemistry", "reaction", "mathematics")  # this one is so interesting.
    # now this one is one to which you may not have a clear answer
    analogize("engineer", "implements", "leader")  # you can learn from the AI.
    analogize("engineer", "implements", "scientist")  # what does scientists do?
    # had to add this after watching "oompa-loompa's of science" bit from BBT.
    analogize("better", "best", "easier")  # syntactic relationship


if __name__ == '__main__':
    main()
