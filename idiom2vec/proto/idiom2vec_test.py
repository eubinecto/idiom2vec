from gensim.models import Word2Vec
from idiom2vec.paths import IDIOM2VEC_WV_001_BIN


def main():
    idiom2vec = Word2Vec.load(IDIOM2VEC_WV_001_BIN)
    idiom = "catch-22"
    print("# --- idioms similar to {} --- #".format(idiom))
    for key, sim in idiom2vec.wv.most_similar(idiom):
        print(key, sim)

    idiom = "beat_around_the_bush"
    print("# --- idioms similar to {} --- #".format(idiom))
    for key, sim in idiom2vec.wv.most_similar(idiom):
        print(key, sim)


if __name__ == '__main__':
    main()
