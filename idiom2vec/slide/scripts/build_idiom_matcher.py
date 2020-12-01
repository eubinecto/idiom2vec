from typing import  List
from config import SLIDE_DIR
from os import path
from spacy import load, Language
from spacy.matcher import Matcher
import pickle
from config import NLP_MODEL, IDIOM_MATCHER_PKL_PATH
from idiom2vec.slide.utils import load_slide_idioms

DELIM = "\t"
SEPARATOR = " "
IDIOM_MIN_WC = 3  # aim for the idioms with length greater than 3
IDIOM_MIN_LENGTH = 14
IDIOM_TOKENIZER_PKL_PATH = path.join(SLIDE_DIR, 'idiom_tokenizer.pkl')

# not to include in the vocabulary
EXCEPTIONS = (
    "if needs be"  # duplicate ->  "if need be" is enough.
)

# placeholder's for possessive pronouns should not be tokenized
POSS_HOLDER_CASES = {
    "one's": [{"ORTH": "one's"}],
    "someone's": [{"ORTH": "someone's"}]
}

SPECIAL_IDIOM_CASES = {
    "catch-22": [{"ORTH": "catch"}, {"ORTH": "-"}, {"ORTH": "22"}]
}


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


def build_idiom_matcher(nlp: Language, idioms: List[str]) -> Matcher:
    """
    uses nlp to build patterns for the matcher.
    """
    global POSS_HOLDER_CASES
    matcher = Matcher(nlp.vocab)  # matcher to build
    # then add idiom matches
    for idiom in idioms:
        # for each idiom, you want to build this.
        patterns: List[List[dict]]
        idiom_doc = nlp(idiom.lower())  # as for building patterns, use uncased version. of the idiom.
        if "-" in idiom:
            # should include both hyphenated & non-hyphenated forms
            # e.g. catch-22, catch 22
            pattern_hyphen = [
                {"TAG": "PRP$"} if token.text in POSS_HOLDER_CASES.keys()
                else {"ORTH": token.text}  # don't use lemma
                for token in idiom_doc
            ]  # include hyphens
            pattern_no_hyphen = [
                {"TAG": "PRP$"} if token.text in POSS_HOLDER_CASES.keys()
                else {"ORTH": token.text}  # don't use lemma
                for token in idiom_doc
                if token.text != "-"
            ]
            patterns = [
                # include two patterns
                pattern_hyphen,
                pattern_no_hyphen
            ]
        else:
            pattern = [
                {"TAG": "PRP$"} if token.text in POSS_HOLDER_CASES.keys()
                else {"LEMMA": token.lemma_}  # if not a verb, we do exact-match
                for token in idiom_doc
            ]
            patterns = [pattern]
        print(patterns)
        matcher.add(idiom, patterns)
    else:
        return matcher


def main():
    global DELIM, IDIOM_MIN_WC, POSS_HOLDER_CASES, SPECIAL_IDIOM_CASES
    # this is the end goal
    # load idioms on to memory.
    idioms = [
        idiom
        for idiom in load_slide_idioms()
        if is_target(idiom)
    ]
    nlp = load(NLP_MODEL)
    # add cases for place holders
    for placeholder, case in POSS_HOLDER_CASES.items():
        nlp.tokenizer.add_special_case(placeholder, case)
    # add cases for words hyphenated with numbers
    for idiom, case in SPECIAL_IDIOM_CASES.items():
        nlp.tokenizer.add_special_case(idiom, case)
    # build patterns for the idioms into a matcher
    idiom_matcher = build_idiom_matcher(nlp, idioms)
    # save it as pickle binary. (matcher is not JSON-serializable.. this is the only way)
    with open(IDIOM_MATCHER_PKL_PATH, 'wb') as fh:
        fh.write(pickle.dumps(idiom_matcher))


if __name__ == '__main__':
    main()
