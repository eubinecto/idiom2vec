from gensim.models import KeyedVectors
from idiom2vec.paths import IDIONLY2VEC_001_KV


def main():
    idionly2vec_kv = KeyedVectors.load_word2vec_format(IDIONLY2VEC_001_KV, binary=False)
    idiom = "catch-22"
    print("# --- idioms similar to {} --- #".format(idiom))
    for key, sim in idionly2vec_kv.most_similar(idiom):
        print(key, sim)

    idiom = "beat_around_the_bush"
    print("# --- idioms similar to {} --- #".format(idiom))
    for key, sim in idionly2vec_kv.most_similar(idiom):
        print(key, sim)


if __name__ == '__main__':
    main()
