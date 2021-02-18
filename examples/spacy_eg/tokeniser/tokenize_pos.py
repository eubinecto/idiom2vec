from spacy import load
from spacy.matcher import Matcher
from spacy.tokens.token import Token
from termcolor import colored
from playground.spacy_pg.tokeniser.tokenize_hardcode import tokenize_with_match


def main():
    sentences = """Like, I'm not saying this of my own accord.
    They made arguments of their own accord, without any evidence.
    The reporter broke the news of his own accord.
    Are you pulling the wool over my eyes?
    He doesn't have any special powers – he's just trying to pull the wool over your eyes.""" \
        .split("\n")

    nlp = load('en_core_web_sm')
    # matcher to use
    matcher = Matcher(nlp.vocab)
    # patterns for "of my own accord".
    patterns_1 = [
        # this is not a phrase matcher, so have to define each token.
        # https://spacy.io/api/annotation#pos-tagging
        [{"LEMMA": "of"},
         # PRP$ is a tag for "pronoun, possessive". (POS=DET)
         # e.g. her, his, its, their, my, etc.
         # for more details: https://spacy.io/api/annotation#pos-tagging
         {"TAG": "PRP$"},
         {"LEMMA": "own"}, {"LEMMA": "accord"}]
    ]
    # note: from 2.2.2 and onwards, the second argument is patterns (not on_match),
    # and callback is now an optional argument.
    matcher.add("of one's own accord", patterns=patterns_1, on_match=None)

    # patterns for idiom:  "pull the wool over one's eyes"
    patterns_2 = [
        [{"LEMMA": "pull"}, {"LEMMA": "the"},
         {"LEMMA": "wool"}, {"LEMMA": "over"},
         {"TAG": "PRP$"}, {"LEMMA": "eye"}]
    ]
    matcher.add("pull the wool over one's eyes", patterns=patterns_2, on_match=None)
    # generate documents
    docs = tokenize_with_match(nlp, matcher, sentences)

    # print out the results
    for doc, sent in zip(docs, sentences):
        print(colored(sent.strip(), 'magenta'))
        for token in doc:
            token: Token
            print("{} --> {}|{}|{}".format(token.text,
                                           colored(token.lemma_, 'blue'),
                                           token.pos_,
                                           token.lex_id))
        print("#############")


if __name__ == '__main__':
    main()
