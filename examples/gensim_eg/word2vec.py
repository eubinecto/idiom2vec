from typing import List, Generator

from gensim.test.utils import datapath
from gensim import utils
from gensim import models
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self) -> Generator[List[str], None, None]:
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            # change this to a list of lemma ids!
            # you can then decode it later!
            yield utils.simple_preprocess(line)


def main():
    corpus = MyCorpus()
    word2vec = models.Word2Vec(sentences=corpus)
    king = word2vec.wv['king']


if __name__ == '__main__':
    main()
