from typing import Generator
from config import SAMPLE_1_NDJSON_PATH, SAMPLE_2_NDJSON_PATH, SAMPLE_3_NDJSON_PATH
import json


SAMPLE_PATHS = (
    SAMPLE_1_NDJSON_PATH,
    SAMPLE_2_NDJSON_PATH,
    SAMPLE_3_NDJSON_PATH
)


def load_examples() -> Generator[dict, None, None]:
    global SAMPLE_PATHS
    for sample_path in SAMPLE_PATHS:
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



