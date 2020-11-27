import csv
from idiom2vec.slide.utils.idiom_nlp import IdiomNLP
from config import IDIOM_MATCHER_PATTERNS_TSV_PATH
import json
DELIM = "\t"


def main():
    global DELIM
    # load idiom matcher from cache.
    idiom_matcher = IdiomNLP.load_idiom_matcher()
    # how do I view the rules..?
    with open(IDIOM_MATCHER_PATTERNS_TSV_PATH, 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter=DELIM)
        # write the header
        tsv_writer.writerow(['idiom', 'pattern'])
        # write the patterns
        for vocab_id, pattern in idiom_matcher._patterns.items():
            idiom = idiom_matcher.vocab.strings[vocab_id]
            tsv_writer.writerow([idiom, json.dumps(pattern)])


if __name__ == '__main__':
    main()
