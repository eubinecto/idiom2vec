# tutorial excerpted from the book: NLP and computational linguistics: a practical guide (2018)
# this script explains how we could vectorize a document into two forms:
# a bow (bag-of-words) vector, or
# a tf-idf vector.

from typing import List
from gensim import corpora, models
import spacy
import logging
from sys import stdout
from termcolor import colored
from os import path
from config import GENSIM_PG_DIR
logging.basicConfig(stream=stdout, level=logging.INFO)
# Here, we make sure that all strings are unicode strings so that
# we can use spaCy for preprocessing

documents = u"""How did this happen to me?
You made a deal, you stupid son of a bitch.
You made a deal with Malebolgia.
You cut a deal for your soul. Cut a deal.
The deal was you'd see Wanda and then become Hellspawn... a ranking officer in the devil's army.
Well, now you've seen her.
Time to pay the piper.""".split("\n")


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


def to_bow(dictionary: corpora.Dictionary,
           preproc_documents: List[List[str]]) -> List[List[tuple]]:
    # first, we need building a dictionary of the vocabulary.
    # this assigns a unique integer to each word
    # Build corpus
    corpus = [
        dictionary.doc2bow(document=preproc_doc)
        for preproc_doc in preproc_documents
    ]
    return corpus


def to_tfidf(corpus: corpora.MmCorpus) -> models.TfidfModel:
    # bow -> tfidf transformation is made easy with gensim.
    # tfidf **trained** on the given corpus.
    # going through each doc, and going through all docs once.
    tfidf = models.TfidfModel(corpus)
    return tfidf


def main():
    global documents
    print(colored("preprocessing docs", 'blue'))
    preproc_documents = preproc_docs()
    for doc, preproc_doc in zip(documents, preproc_documents):
        print(doc)
        print(preproc_doc)
        print("----")

    print(colored("build dictionary from tokenized documents", 'blue'))
    dictionary = corpora.Dictionary(preproc_documents)
    print(dictionary)
    for key, val in dictionary.items():
        # each integer corresponds to a word in the dictionary (vocab)
        print(str(key) + " -> " + val)

    print(colored("vectorisation (BOW) with gensim", 'blue'))
    corpus = to_bow(dictionary, preproc_documents)
    for doc_as_bow in corpus:
        # use dictionary again to look up words by their unique ids
        doc = [dictionary.get(word_id) for word_id, _ in doc_as_bow]
        bow_vec = [word_cnt for _, word_cnt in doc_as_bow]
        print("doc:", doc)
        print("bow_vec:", bow_vec)
        print("----")

    # now our corpus is assembled!
    # what if corpus is too huge? we can write the corpus in disk, and
    # let gensim use "streaming corpus"
    # MmCorpus -> for serializing corpus to sparse Matrix Market format
    corpus_path = path.join(GENSIM_PG_DIR, "corpus.mm")
    corpora.MmCorpus.serialize(corpus_path,
                               # documentation states: corpus in bow format.
                               corpus=corpus)

    # you can later load the serialised corpus from disk, and this will be far more
    # memory efficient.
    # the output below should be the same as that of above
    corpus = corpora.MmCorpus(corpus_path)

    print(colored("what does a tf-idf representation look like?"))
    tfidf_model = to_tfidf(corpus)  # using the corpus loaded from disk.
    # class TfidfModel supports __getitem__
    # parameter type from the doc: bow{list of (int, int), iterable of iterable of (int, int)}
    # but I'm not likely to use this.
    for doc_as_bow in corpus:
        doc_as_tfidf = tfidf_model[doc_as_bow]
        doc = [dictionary.get(word_id) for word_id, _ in doc_as_bow]
        tfidf_vec = [tfidf_score for _, tfidf_score in doc_as_tfidf]
        print("doc:", doc)
        print("tfidf_vec:", tfidf_vec)
        print("----")


if __name__ == '__main__':
    main()
