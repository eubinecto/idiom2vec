# tokenisation with spacy
# + building my own tokeniser.
# how do I do this?
# 1. build a matcher that matches idiomatic phrases
# 2. merge each phrase using retokenize

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens.token import Token
sentences = """Somebody's gotta pay the piper.
Hey, you mess around, you got to pay the piper.
But now the time has come to pay the piper.
Like, I'm not saying this of my own accord.
She made arguments of her own accord, without any evidence."""\
.split("\n")

nlp = spacy.load(name='en_core_web_sm')
# matcher to use
matcher = PhraseMatcher(nlp.vocab)
# define the key, and the documents that should match with the key
matcher.add("pay the piper", None,
            nlp("pay the piper"))
matcher.add("of one's own accord", None,
            # include all possible substitutes for "own"
            nlp("of my own accord"), nlp("of his own accord"),
            nlp("of their own accord"), nlp("of her own accord"))


def main():
    global sentences, nlp, matcher
    docs = [nlp(sent) for sent in sentences]
    for doc in docs:
        matches = matcher(doc)
        # PhraseMatcher.__call__ returns a list of (match_id, start, end) tuples.
        for match_id, start, end in matches:
            # get back the string representation of the match
            match_id_string = nlp.vocab.strings[match_id]
            with doc.retokenize() as retokenizer:
                retokenizer.merge(doc[start:end],
                                  # use match_id_string to be the representative
                                  attrs={"LEMMA": match_id_string})
        for token in doc:
            token: Token
            print("text:{}|lemma:{}".format(token.text, token.lemma_))
        print("----")


if __name__ == '__main__':
    main()
