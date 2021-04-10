from gensim.models import Word2Vec
from idiom2vec.configs import IDIOM2VEC_001_MODEL


def main():
    idiom2vec_model = Word2Vec.load(str(IDIOM2VEC_001_MODEL))

    # anything similar to catch-22?
    print("###catch-22")
    for word, score in idiom2vec_model.wv.similar_by_word('catch-22', topn=30):
        print(word, score)

    print("\n###as long as")
    for word, score in idiom2vec_model.wv.similar_by_word('as long as', topn=30):
        print(word, score)

    print("\n###you name it")
    for word, score in idiom2vec_model.wv.similar_by_word('you name it', topn=30):
        print(word, score)


if __name__ == '__main__':
    main()
