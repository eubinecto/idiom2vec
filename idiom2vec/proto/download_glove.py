import gensim.downloader as api
from gensim.models import KeyedVectors


def main():
    glove: KeyedVectors = api.load("glove-wiki-gigaword-200")
    # save glove as a binary file.
    glove.save_word2vec_format("./glove", binary=True)


if __name__ == '__main__':
    main()
