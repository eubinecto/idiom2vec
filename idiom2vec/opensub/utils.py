from typing import Generator, List
from config import NDJSON_SAMPLES_PATH, SUBS_PATH
import json

from idiom2vec.slide.utils import IdiomNLP


# load functions for training from samples
def load_examples() -> Generator[dict, None, None]:
    for sample_path in NDJSON_SAMPLES_PATH:
        with open(sample_path, 'r') as fh:
            for line in fh:
                yield json.loads(line)


def cleanse_resp(resp: str) -> str:
    # e.g. You can't-- Ahh!
    return resp.replace("--", " ")


# use load_sample above to get only the responses.
def load_resps() -> Generator[str, None, None]:
    return (
        cleanse_resp(example['response'])
        for example in load_examples()
    )


# load functions for training from the original data
def load_subs() -> Generator[str, None, None]:
    with open(SUBS_PATH, 'r') as fh:
        for sub in fh:
            yield sub


def load_subs_tokenized() -> Generator[List[str], None, None]:
    pass


class OpenSubCorpus:
    def __init__(self, idiom_nlp: IdiomNLP):
        self.idiom_nlp = idiom_nlp

    def __iter__(self) -> Generator[List[str], None, None]:
        return load_subs_tokenized()
