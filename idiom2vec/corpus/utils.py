from typing import Generator
from config import SAMPLE_SUB_NDJSON_PATH
import json


def load_sample_sub_ndjson() -> Generator[dict, None, None]:
    with open(SAMPLE_SUB_NDJSON_PATH, 'r') as fh:
        for line in fh:
            yield json.loads(line)
