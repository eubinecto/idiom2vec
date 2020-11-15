# tutorial excerpted from the book: NLP and computational linguistics: a practical guide (2018)
from typing import List

from gensim import corpora
import spacy
import logging
from sys import stdout
from termcolor import colored
logging.basicConfig(stream=stdout, level=logging.INFO)
# Here, we make sure that all strings are unicode strings so that
# we can use spaCy for preprocessing
documents = [
    u"Football club Arsenal defeat local rivals this weekend.",
    u"Weekend football frenzy takes over London.", u"Bank open for takeover bids after losing millions.",
    u"London football clubs bid to move to Wembley stadium.",
    u"Arsenal bid 50 million pounds for striker Kane.",
    u"Financial troubles result in loss of millions for bank.",
    u"Western bank files for bankruptcy after financial losses.",
    u"London football club is taken over by oil millionaire from Russia.",
    u"Banking on finances not working for Russia."
]  # this is the corpus to use


def preproc_docs() -> list:
    global documents
    # what is this function?
    # before you do this, make sure you've downloaded the model you'd like to load
    # python3 -m spacy download en_core_web_sm
    # this will load a "nlp" model
    nlp = spacy.load(name='en_core_web_sm')
    texts = list()
    for document in documents:
        text = list()
        # the central datastructures in spaCy is Doc and Vocab.
        # here, calling nlp will give an instance of Doc.
        doc = nlp(document)
        for w in doc:
            # skip stop words, punctuations, ...like_num? like numbers?
            if not w.is_stop and not w.is_punct and not w.like_num:
                # this is how you do lemmatisation with spacy!
                text.append(w.lemma_)
        else:
            texts.append(text)
    else:
        return texts


def to_bow(preproc_documents: List[List[str]]) -> List[List[tuple]]:
    # first, we need building a dictionary of the vocabulary.
    # this assigns a unique integer to each word
    dictionary = corpora.Dictionary(preproc_documents)
    # Build corpus
    corpus = [
        dictionary.doc2bow(document=preproc_doc)
        for preproc_doc in preproc_documents
    ]
    return corpus


def main():
    global documents
    print(colored("preprocessing docs", 'blue'))
    preproc_documents = preproc_docs()
    for doc, preproc_doc in zip(documents, preproc_documents):
        print(doc)
        print(preproc_doc)
        print("----")

    print(colored("vectorisation (BOW) with gensim", 'blue'))
    corpus = to_bow(preproc_documents)
    for bow, preproc_doc in zip(corpus, preproc_documents):
        print(preproc_doc)
        print(bow)
        print("----")


if __name__ == '__main__':
    main()
