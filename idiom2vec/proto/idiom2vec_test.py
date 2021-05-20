from gensim.models import Word2Vec
from idiom2vec.paths import IDIOM2VEC_WV_001_BIN, IDIOM2VEC_WV_002_BIN, IDIOM2VEC_WV_003_BIN, IDIOM2VEC_WV_004_BIN


def sim_to(idiom: str, idiom2vec: Word2Vec):
    print("# --- words similar to {} --- #".format(idiom))
    for key, sim in idiom2vec.wv.most_similar(idiom, topn=20):
        print(key, sim)


def main():
    idiom2vec_001 = Word2Vec.load(IDIOM2VEC_WV_001_BIN)
    idiom2vec_002 = Word2Vec.load(IDIOM2VEC_WV_002_BIN)
    idiom2vec_003 = Word2Vec.load(IDIOM2VEC_WV_003_BIN)
    idiom2vec_004 = Word2Vec.load(IDIOM2VEC_WV_004_BIN)
    print(idiom2vec_002.wv.get_vector("with_bated_breath"))
    sim_to("in_the_zone", idiom2vec_001)
    sim_to("in_the_zone", idiom2vec_002)
    sim_to("in_the_zone", idiom2vec_003)
    sim_to("in_the_zone", idiom2vec_004)


if __name__ == '__main__':
    main()
