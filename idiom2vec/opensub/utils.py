from typing import Generator
from config import NDJSON_SAMPLES_PATH
import json


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



