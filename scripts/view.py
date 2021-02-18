from idiom2vec.config import IDIOM2VEC_PKL_PATH
from gensim import models


def main():
    idiom2vec = models.Word2Vec.load(IDIOM2VEC_PKL_PATH)
    for token in idiom2vec.wv.index2entity:
        print(token)


if __name__ == '__main__':
   main()



