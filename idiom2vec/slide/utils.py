import pickle

from spacy import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc
from config import IDIOM_MATCHER_PKL_PATH
from os import path


class MergeIdiomComponent:
    def __init__(self, idiom_matcher):
        self.idiom_matcher = idiom_matcher

    def __call__(self, doc: Doc) -> Doc:
        matches = self.idiom_matcher(doc)
        for match_id, start, end in matches:
            # get back the lemma for this match
            # note: matcher has references to the vocab on its own!
            match_lemma = self.idiom_matcher.vocab.strings[match_id]
            # retokenise
            with doc.retokenize() as retokeniser:
                retokeniser.merge(doc[start:end],
                                  # set tag as the idiom
                                  attrs={'LEMMA': match_lemma, 'TAG': 'IDIOM'})
        return doc


# factory method for the component
def create_merge_idiom_component(nlp, name, idiom_matcher) -> MergeIdiomComponent:
    if not idiom_matcher:
        raise ValueError("idiom_matcher does not exist.")
    return MergeIdiomComponent(idiom_matcher)


class IdiomNLP:
    def __init__(self, nlp):
        self.nlp = nlp
        self.idiom_matcher = IdiomNLP.load_idiom_matcher()
        # build a factory with the matcher
        Language.factory(
            name="merge_idiom",
            retokenizes=True,
            default_config={"idiom_matcher": self.idiom_matcher},
            func=create_merge_idiom_component
        )
        # add the pipe, and save it to disk
        nlp.add_pipe("merge_idiom", after="lemmatizer")

    def __call__(self, text: str, *args, **kwargs):
        # just a wrapper, to construct the pipeline on init.
        return self.nlp(text)

    @staticmethod
    def load_idiom_matcher() -> Matcher:
        if not path.exists(IDIOM_MATCHER_PKL_PATH):
            raise ValueError
        with open(IDIOM_MATCHER_PKL_PATH, 'rb') as fh:
            return pickle.loads(fh.read())
