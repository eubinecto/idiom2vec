from merge_idioms.builders import MIPBuilder
from idiom2vec.utils import OpenSubCorpus
from idiom2vec.config import IDIOM2VEC_PKL_PATH
import logging
import sys
import gensim
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    # prepare the corpus
    mip_builder = MIPBuilder()
    mip_builder.construct()
    opensub_corpus = OpenSubCorpus(mip_builder.mip)
    # start training.. well you don't see any progress.. why?
    word2vec = gensim.models.Word2Vec(sentences=opensub_corpus, workers=10)
    print("training done")
    word2vec.save(IDIOM2VEC_PKL_PATH)


if __name__ == '__main__':
    main()
