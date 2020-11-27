import spacy
from spacy.tokens import Doc


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # this is where tokenisation occurs..
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)  # True * word count.
        return Doc(self.vocab, words=words, spaces=spaces)


def main():
    nlp = spacy.load("en_core_web_sm")
    # overwrite the tokeniser for nlp with ours
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    doc = nlp("What's happened to me? he thought. It wasn't a dream.")
    print([t.text for t in doc])


if __name__ == '__main__':
    main()
