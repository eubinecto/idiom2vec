"""
trains an idiom2vec model.
"""
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec
from idiom2vec.corpora import IdiomSentences
from idiom2vec.paths import (
    IDIOM2VEC_WV_001_BIN,
    IDIOM2VEC_WV_002_BIN,
    IDIOM2VEC_DV_001_BIN, IDIOM2VEC_WV_003_BIN, GLOVE, IDIOM2VEC_WV_004_BIN
)
from sys import stdout
from matplotlib import pyplot as plt
import argparse
import logging
import gensim.downloader as api
# projection weights... right...?
logging.basicConfig(stream=stdout, level=logging.INFO)


# to be used for printing out the loss
class Idiom2VecCallback(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0
        self.losses = list()  # collect losses here
        self.losses.append(0.0)

    def on_epoch_end(self, model):
        """
        this will be called at the end of each epoch.
        """
        loss = model.get_latest_training_loss()
        self.epoch += 1
        self.losses.append(loss)
        print('Cumulative loss after epoch {}: {}'.format(self.epoch, loss))
        print('Offset to previous loss: {}'.format(str(self.losses[-1] - self.losses[-2])))
        # self.visualise_loss()  # plot the loss   # this would probably not work in command line. in the server.

    def visualise_loss(self):
        # x = epoch
        # y = cumulative loss
        x = list(range(1, len(self.losses) + 1))
        plt.plot(x, self.losses)
        plt.xlabel("epoch")
        plt.ylabel("the cumulative log loss")
        # set both axes to be logarithmic.
        plt.show()


def train_with_word2vec(args,  sents: IdiomSentences):

    # -- params set up --- #
    w2v_params = {
        'vector_size': args.vector_size,
        'window': args.window,
        'alpha': args.alpha,
        'min_count': args.min_count,
        'workers': args.workers,
        'sg': args.sg,
        'epochs': args.epochs,
        'compute_loss': args.compute_loss
    }

    # --- paths setup --- #
    if args.model_version == "001":
        idiom2vec_bin_path = IDIOM2VEC_WV_001_BIN
    elif args.model_version == "002":
        idiom2vec_bin_path = IDIOM2VEC_WV_002_BIN
    elif args.model_version == "003":
        idiom2vec_bin_path = IDIOM2VEC_WV_003_BIN
    elif args.model_version == "004":
        idiom2vec_bin_path = IDIOM2VEC_WV_004_BIN
    else:
        raise ValueError

    # --- logger setup --- #
    logger = logging.getLogger("train_idiom2vec")
    if args.intersect_glove:
        idiom2vec = Word2Vec(**w2v_params)
        # intersect glove. Do not change the vectors.
        idiom2vec.wv.intersect_word2vec_format(GLOVE, binary=True, lockf=0.0)
        idiom2vec.build_vocab(sents)  # then build a vocab.
        # --- training a word2vec model from scratch --- #
        idiom2vec.train(sents,
                        epochs=idiom2vec.epochs,
                        compute_loss=idiom2vec.compute_loss,
                        total_examples=idiom2vec.corpus_count,
                        callbacks=[Idiom2VecCallback()])
    else:
        # train the model
        idiom2vec = Word2Vec(**w2v_params,
                             sentences=sents,
                             callbacks=[Idiom2VecCallback()])

    # --- save the model --- #
    idiom2vec.save(idiom2vec_bin_path)


def train_with_doc2vec(args, sents: IdiomSentences):
    # --- params setup --- #
    d2v_params = {
        'vector_size': args.vector_size,
        'window': args.window,
        'alpha': args.alpha,
        'min_count': args.min_count,
        'workers': args.workers,
        'epochs': args.epochs,
        'compute_loss': args.compute_loss,
        'dm_concat': args.dm_concat,
        # if 1, it trains word vectors simultaneously with doc vectors.
        'dbow_words': args.dbow_words
    }
    # documents setup --- #
    docs = [
        TaggedDocument(sent, "sent_" + str(idx))
        for idx, sent in enumerate(sents)
    ]

    # --- paths setup --- #
    if args.model_version == "001":
        idiom2vec_bin_path = IDIOM2VEC_DV_001_BIN
    else:
        raise ValueError
    idiom2vec = Doc2Vec(**d2v_params,
                        documents=docs,
                        callbacks=[Idiom2VecCallback()])

    idiom2vec.save(idiom2vec_bin_path)


def main():
    parser = argparse.ArgumentParser()
    # the size of the embedding vector.
    parser.add_argument('--vector_size',
                        type=int,
                        default=200)
    # the size of the window
    parser.add_argument('--window',
                        type=int,
                        default=10)
    # only the words with a frequency above the minimum count will be included in the vocab.
    parser.add_argument('--min_count',
                        type=int,
                        default=5)
    parser.add_argument('--alpha',
                        type=float,
                        default=0.025)
    # number of workers to use for training word2vec
    parser.add_argument('--workers',
                        type=int,
                        default=4)
    # 1: use skipgram. 0: use cbow.
    parser.add_argument('--sg',
                        type=int,
                        default=1)
    # number of epochs.
    parser.add_argument('--epochs',
                        type=int,
                        default=100)
    # compute loss
    parser.add_argument('--compute_loss',
                        dest='compute_loss',
                        default=False,
                        action='store_true')
    parser.add_argument('--remove_stopwords',
                        dest='remove_stopwords',
                        default=False,
                        action='store_true')
    parser.add_argument('--remove_propns',
                        dest='remove_propns',
                        default=False,
                        action='store_true')
    parser.add_argument('--intersect_glove',
                        dest='intersect_glove',
                        default=False,
                        action='store_true')
    parser.add_argument('--dm_concat',
                        type=int,
                        default=0)
    parser.add_argument('--dbow_words',
                        type=int,
                        default=1)
    # the files to save
    parser.add_argument('--model_version',
                        type=str,
                        default="002")
    parser.add_argument('--train_with',
                        type=str,
                        default="word2vec")
    args = parser.parse_args()
    # --- prepare the corpus --- #
    # the corpus that will be streamed
    sents = IdiomSentences(args.remove_stopwords, args.remove_propns)

    # --- start training --- #
    if args.train_with == "word2vec":
        train_with_word2vec(args, sents)
    elif args.train_with == "doc2vec":
        train_with_doc2vec(args, sents)
    else:
        raise ValueError


if __name__ == '__main__':
    main()
