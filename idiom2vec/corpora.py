"""
Corpora to be used for training.
"""
import json
from typing import Generator, List
from spacy.lang.en import STOP_WORDS
from idiom2vec.paths import IDIOM2LEMMA2POS_TSV
import csv


class IdiomSentences:

    def __init__(self,
                 remove_stopwords: bool = False,
                 remove_propns: bool = False):
        self.remove_stopwords = remove_stopwords
        self.remove_propns = remove_propns

    def __iter__(self) -> Generator[List[str], None, None]:
        """
        https://stackoverflow.com/a/40447086
        gensim's word2vec automatically down-samples all frequent words
        """
        with open(IDIOM2LEMMA2POS_TSV, 'r') as fh:
            tsv_reader = csv.reader(fh, delimiter="\t")
            for row in tsv_reader:
                idiom = row[0]
                lemma2pos = json.loads(row[1])
                # insert the idiom back to the sentence
                # just get the lemmas
                idiom_idx = [lemma for lemma, _ in lemma2pos].index("[IDIOM]")
                lemma2pos[idiom_idx] = (idiom, "X")  # make sure to put x.
                if self.remove_stopwords:
                    lemma2pos = [
                        (lemma, pos)
                        for lemma, pos in lemma2pos
                        if lemma not in STOP_WORDS
                    ]
                if self.remove_propns:
                    lemma2pos = [
                        (lemma, pos)
                        for lemma, pos in lemma2pos
                        if pos != "PROPN"
                    ]
                lemmas = [lemma for lemma, _ in lemma2pos]
                yield lemmas
