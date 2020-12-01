from typing import List
import csv
import json
from wiktionaryparser import WiktionaryParser
from config import SLIDE_POS_TSV_PATH
from idiom2vec.slide.utils import load_slide_idioms

WIK_PARSER = WiktionaryParser()
HEADER = [
    "idiom",
    "pos"
]


def dl_idiom_pos(idiom: str) -> List[str]:
    global WIK_PARSER
    res = WIK_PARSER.fetch(idiom)
    return [
        word_def['partOfSpeech']
        for word in res
        for word_def in word['definitions']
    ]


def main():
    global HEADER
    idioms = load_slide_idioms()
    with open(SLIDE_POS_TSV_PATH, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        tsv_writer.writerow(HEADER)
        for idiom in idioms:
            # note that this is a list of pos
            pos = dl_idiom_pos(idiom)
            to_write = [idiom, json.dumps(pos)]
            tsv_writer.writerow(to_write)
            print(to_write)


if __name__ == '__main__':
    main()

