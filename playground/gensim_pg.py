
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
import numpy as np
from config import FASTTEXT_VEC_PATH, FASTTEXT_PKL_PATH
import logging
import sys
from os import path
import pickle
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def load_fasttext_model() -> Word2VecKeyedVectors:
    logger = logging.getLogger("load_fasttext_model")
    """
    load the model from cache if stored already. if not, build the model. (will take about 10 minutes)
    :return:
    """
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


def main():
    """
    play ground for getting familiar with gensim library. It is indeed made for humans!
    :return:
    """
    # load fasttext word2vec model
    fasttext_model = load_fasttext_model()
    # you can compute one-to-one dist between two lexicons (cosine distance)
    d1: float = fasttext_model.distance("cat", "dog")
    # note that lexvec is lower-cased. you should lower strings before you use it.
    d2: float = fasttext_model.distance("cat", "Human".lower())
    print(d1, d2)
    # or, one-to-many dists (cosine dists)
    dists = fasttext_model.distances("cat", ["dog", "human"])
    print(dists)
    # if you want to find dist with measures other than cosine (e.g. l2, l2, dot product, vector addition, etc).
    # you'll have to get the vectors directly compute the distances yourself.
    # they return numpy nd arrays
    zebra: np.ndarray = fasttext_model.get_vector("zebra")
    stripe: np.ndarray = fasttext_model.get_vector("stripe")
    brown: np.ndarray = fasttext_model.get_vector("brown")
    # you can do vector operations
    sims = fasttext_model.similar_by_vector(zebra - stripe + brown, topn=10)
    # you will get antelope in one of the results!
    print(sims)


if __name__ == '__main__':
    main()
