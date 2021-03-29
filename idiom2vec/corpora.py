import json
from typing import Generator, List

from idiom2vec.configs import SPOK_2017_TRAIN_NDJSON


class Spok2017:
    """
    This is the corpus to be used for prototyping.
    a returns a generator.
    the generator streams out a list of tokens.
    """
    def __iter__(self) -> Generator[List[str], None, None]:
        with open(SPOK_2017_TRAIN_NDJSON, 'r') as fh:
            for line in fh:
                tokenised = json.loads(line)
                yield tokenised
