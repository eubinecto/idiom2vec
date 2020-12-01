from typing import Generator
from config import SAMPLE_SUB_NDJSON_PATH
import json


def load_sample() -> Generator[dict, None, None]:
    with open(SAMPLE_SUB_NDJSON_PATH, 'r') as fh:
        for line in fh:
            yield json.loads(line)


def cleanse_resp(resp: str) -> str:
    # e.g. You can't-- Ahh!
    return resp.replace("--", " ")


# use load_sample above to get only the responses.
def load_resps() -> Generator[str, None, None]:
    return (
        cleanse_resp(example['response'])
        for example in load_sample()
    )



