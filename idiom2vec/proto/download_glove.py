import gensim.downloader as api
from gensim.models import KeyedVectors


def main():
    glove: KeyedVectors = api.load("glove-wiki-gigaword-200")
    # save glove as a binary file.
    for word, score in glove.similar_by_word('focus'):
        print(word, score)


if __name__ == '__main__':
    main()
