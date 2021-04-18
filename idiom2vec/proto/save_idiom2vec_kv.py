from gensim.models import Word2Vec

from idiom2vec.paths import IDIOM2VEC_WV_001_BIN, IDIOM2VEC_001_KV


def main():
    idiom2vec = Word2Vec.load(IDIOM2VEC_WV_001_BIN)
    idiom2vec.wv.save_word2vec_format(IDIOM2VEC_001_KV, binary=True)


if __name__ == '__main__':
    main()
