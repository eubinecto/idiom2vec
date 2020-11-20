from spacy import load
from spacy.matcher import Matcher
from spacy.tokens.token import Token
from termcolor import colored
from playground.spacy_pg.tokenize_hardcode import tokenize_with_match


def main():
    sentences = """Like, I'm not saying this of my own accord.
    They made arguments of their own accord, without any evidence.
    The reporter broke the news of his own accord.""" \
        .split("\n")

    nlp = load('en_core_web_sm')
    # matcher to use
    matcher = Matcher(nlp.vocab)
    patterns = [
        # this is not a phrase matcher, so have to define each token.
        # https://spacy.io/api/annotation#pos-tagging
        [{"LOWER": "of"},
         # PRP$ is a tag for "pronoun, possessive". (POS=DET)
         # e.g. her, his, its, their, my, etc.
         {"TAG": "PRP$"},
         {"LOWER": "own"}, {"LOWER": "accord"}]
    ]
    # note: from 2.2.2 and onwards, the second argument is patterns (not on_match),
    # and callback is now an optional argument.
    matcher.add("of one's own accord", patterns=patterns, on_match=None)
    docs = tokenize_with_match(nlp, matcher, sentences)
    for doc in docs:
        for token in doc:
            token: Token
            print("{} --> {}|{}|{}".format(token.text,
                                           colored(token.lemma_, 'blue'),
                                           token.pos_,
                                           token.lex_id))
        print("#############")


if __name__ == '__main__':
    main()
