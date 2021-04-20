from gensim.models import Word2Vec
from idiom2vec.paths import IDIOM2VEC_WV_001_BIN, IDIOM2VEC_WV_002_BIN


def sim_to(idiom: str, idiom2vec: Word2Vec):
    print("# --- words similar to {} --- #".format(idiom))
    for key, sim in idiom2vec.wv.most_similar(idiom):
        print(key, sim)


def main():
    idiom2vec_001 = Word2Vec.load(IDIOM2VEC_WV_001_BIN)
    idiom2vec_002 = Word2Vec.load(IDIOM2VEC_WV_002_BIN)

    sim_to("beat_around_the_bush", idiom2vec_001)
    sim_to("beat_around_the_bush", idiom2vec_002)


if __name__ == '__main__':
    main()
