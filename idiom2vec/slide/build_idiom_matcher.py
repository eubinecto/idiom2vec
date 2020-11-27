import json
from typing import Generator, List

from spacy.tokens import Doc

from config import SLIDE_DIR
from os import path
import csv
from spacy import load, Language
from spacy.matcher import Matcher
import pickle
from config import NLP_MODEL, IDIOM_MATCHER_PKL_PATH

DELIM = "\t"
SEPARATOR = " "
IDIOM_MIN_WC = 3  # aim for the idioms with length greater than 3
IDIOM_MIN_LENGTH = 14
SLIDE_TSV_PATH = path.join(SLIDE_DIR, "slide.tsv")
IDIOM_TOKENIZER_PKL_PATH = path.join(SLIDE_DIR, 'idiom_tokenizer.pkl')

# not to include in the vocabulary
EXCEPTIONS = (
    "if needs be"  # duplicate. Use if need be.
)

# placeholder's for possessive pronouns should not be tokenized
POSS_HOLDER_CASES = {
    "one's": [{"ORTH": "one's"}],
    "someone's": [{"ORTH": "someone's"}]
}


# ------ for preprocessing data --------- #
def load_slide_idioms() -> Generator[str, None, None]:
    global SLIDE_TSV_PATH, DELIM
    with open(SLIDE_TSV_PATH, 'r') as fh:
        slide_tsv = csv.reader(fh, delimiter=DELIM)
        # skip the  header
        next(slide_tsv)
        for row in slide_tsv:
            yield row[0]


def is_above_min_len(idiom: str) -> bool:
    global IDIOM_MIN_LENGTH
    return len(idiom) >= IDIOM_MIN_LENGTH


def is_above_min_wc(idiom: str) -> bool:
    global IDIOM_MIN_WC, SEPARATOR
    return len(idiom.split(SEPARATOR)) >= IDIOM_MIN_WC


def is_hyphenated(idiom: str) -> bool:
    return idiom.find("-") != -1


def is_not_exception(idiom: str) -> bool:
    global EXCEPTIONS
    return idiom not in EXCEPTIONS


def is_target(idiom: str) -> bool:
    # if it is either long enough or hyphenated, then I'm using it.
    return is_not_exception and \
           (is_above_min_wc(idiom) or is_above_min_len(idiom) or is_hyphenated(idiom))


# focus on one thing -> think less!
def cleanse_idiom(idiom: str) -> str:
    return idiom.replace("-", " ")


def build_idiom_matcher(nlp: Language, idioms: List[str]) -> Matcher:
    """
    uses nlp to build patterns for the matcher.
    """
    global POSS_HOLDER_CASES
    matcher = Matcher(nlp.vocab)  # matcher to build
    # then add idiom matches
    for idiom in idioms:
        patterns = list()
        idiom_doc = nlp(idiom)
        # construct a pattern for each idiom..
        for token in idiom_doc:
            if token.text in POSS_HOLDER_CASES.keys():
                pattern = {"TAG": "PRP$"}
            else:
                # use the lemma as the
                pattern = {"LEMMA": token.lemma_}
            patterns.append(pattern)
        else:
            # add the pattern here
            print(patterns)
            matcher.add(idiom_doc.text, [patterns])
    else:
        return matcher


def main():
    global DELIM, IDIOM_MIN_WC, POSS_HOLDER_CASES
    # this is the end goal
    # load idioms on to memory.
    idioms = [
        cleanse_idiom(idiom)
        for idiom in load_slide_idioms()
        if is_target(idiom)
    ]
    nlp = load(NLP_MODEL)
    # add rules for place holders
    for placeholder, case in POSS_HOLDER_CASES.items():
        nlp.tokenizer.add_special_case(placeholder, case)
    # get the matcher
    idiom_matcher = build_idiom_matcher(nlp, idioms)
    # save it as pickle binary. (matcher is not JSON-serializable.. this is the only way)
    with open(IDIOM_MATCHER_PKL_PATH, 'wb') as fh:
        fh.write(pickle.dumps(idiom_matcher))


if __name__ == '__main__':
    main()
