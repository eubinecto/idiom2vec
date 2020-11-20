# tokenisation with spacy
# + building my own tokeniser.
# how do I do this?
# 1. build a matcher that matches idiomatic phrases
# 2. merge each phrase using retokenize
from typing import List, Union
from spacy import load, Language
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens.token import Token
from termcolor import colored


def tokenize_with_match(nlp: Language,
                        matcher: Union[Matcher, PhraseMatcher],
                        sentences: List[str]) -> List[Doc]:
    docs = [nlp(sent.strip()) for sent in sentences]
    for doc in docs:
        matches = matcher(doc)
        # PhraseMatcher.__call__ returns a list of (match_id, start, end) tuples.
        for match_id, start, end in matches:
            # get back the string representation of the match
            match_id_string = nlp.vocab.strings[match_id]
            with doc.retokenize() as retokenizer:
                # merge them.
                retokenizer.merge(doc[start:end],
                                  # use match_id_string to be the representative
                                  attrs={"LEMMA": match_id_string})

    return docs


def main():
    sentences = """Somebody's gotta pay the piper.
    Hey, you mess around, you got to pay the piper.
    But now the time has come to pay the piper.
    Like, I'm not saying this of my own accord.
    She made arguments of her own accord, without any evidence.""" \
        .split("\n")

    nlp = load(name='en_core_web_sm')
    # matcher to use
    matcher = PhraseMatcher(nlp.vocab)
    # define the key, and the documents that should match with the key
    matcher.add("pay the piper", None,
                nlp("pay the piper"))
    matcher.add("of one's own accord", None,
                # include all possible substitutes for "own"
                nlp("of my own accord"), nlp("of his own accord"),
                nlp("of their own accord"), nlp("of her own accord"))
    docs = tokenize_with_match(nlp, matcher, sentences)
    for doc in docs:
        for token in doc:
            token: Token
            print("{} --> {}".format(token.text, colored(token.lemma_, 'blue')))
        print("----")


if __name__ == '__main__':
    main()
