"""
Corpora to be used for training.
"""
import json
from typing import Generator, List
from idiom2vec.paths import IDIOM2SENT_TSV
import csv
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class IdiomSentences:
    lemmatizer = WordNetLemmatizer()
    stopwords_set = set(stopwords.words('english'))

    def __iter__(self) -> Generator[List[str], None, None]:
        """
        https://stackoverflow.com/a/40447086
        gensim's word2vec automatically down-samples all frequent words
        """

        with open(IDIOM2SENT_TSV, 'r') as fh:
            tsv_reader = csv.reader(fh, delimiter="\t")
            for row in tsv_reader:
                idiom = row[0]
                sent = json.loads(row[1])
                idiom_idx = sent.index("[IDIOM]")
                sent[idiom_idx] = idiom
                # lemmatise, and filter out stopwords.
                sent_lemmatised = [
                    self.lemmatizer.lemmatize(token)
                    for token in sent
                ]
                yield sent_lemmatised
