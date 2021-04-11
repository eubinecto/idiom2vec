"""
Corpora to be used for training.
"""
import json
from typing import Generator, List


class Corpus:
    def __init__(self, train_ndjson_path: str, doc_is_sent: bool = True):
        self.train_ndjson_path = train_ndjson_path
        self.doc_is_sent = doc_is_sent

    def __iter__(self):
        raise NotImplementedError


class Coca(Corpus):
    def __iter__(self) -> Generator[List[str], None, None]:
        with open(self.train_ndjson_path, 'r') as fh:
            for line in fh:
                sents = json.loads(line)
                if self.doc_is_sent:
                    # this will be a list of lists.
                    for sent in sents:
                        # at least two
                        if len(sent) > 1:
                            yield sent
                else:
                    # flatten out the list of lists
                    article = [
                        token
                        for sent in sents
                        for token in sent
                        # make sure the token is not empty.
                        if token
                        ]
                    yield article


class Opensub(Corpus):
    """
    open subtitles.
    """
    def __iter__(self):
        pass


class Bns(Corpus):
    """
    british national corpus
    """

    def __iter__(self):
        pass
