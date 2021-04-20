"""
Corpora to be used for training.
"""
import json
from typing import Generator, List
from idiom2vec.paths import IDIOM2LEMMA2POS_TSV
import csv


class IdiomSentences:
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
                # just get the lemmas
                lemmas = [
                    lemma
                    for lemma, pos in lemma2pos
                ]
                idiom_idx = lemmas.index("[IDIOM]")
                lemmas[idiom_idx] = idiom
                yield lemmas
