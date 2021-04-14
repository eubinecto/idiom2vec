"""
trains an idiom2vec model.
"""
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from idiom2vec.corpora import IdiomSentences
from idiom2vec.paths import (
    IDIOM2VEC_001_BIN,
    IDIONLY2VEC_001_KV
)
from identify_idioms.service import load_idioms
from sys import stdout
from matplotlib import pyplot as plt
import argparse
import logging
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
                        default=1)
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

    # the files to save
    parser.add_argument('--model_version',
                        type=str,
                        default="001")

    parser.add_argument('--pretrained_model',
                        type=str,
                        default="glove-wiki-gigaword-200")
    args = parser.parse_args()

    # --- params setup --- #
    # the parameters to pass to word2vec.
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
        idiom2vec_bin_path = IDIOM2VEC_001_BIN
        idionly2vec_kv_path = IDIONLY2VEC_001_KV
    else:
        raise

    # --- logger setup --- #
    logger = logging.getLogger("train_idiom2vec")

    # --- prepare the corpus --- #
    # the corpus that will be streamed
    sents = IdiomSentences()

    # --- pre-load the model --- #
    # train it with idiomatic sentences
    idiom2vec = Word2Vec(**w2v_params,
                         sentences=sents,
                         callbacks=[Idiom2VecCallback()])

    # save idiom2vec.
    idiom2vec.save(idiom2vec_bin_path)

    # --- save idionly2vec --- #
    idioms = [
        idiom
        for idiom in load_idioms()  # get this from identify idioms lib.
        if idiom2vec.wv.key_to_index.get(idiom, None)
    ]
    idiom_vectors = [
        idiom2vec.wv.get_vector(idiom)
        for idiom in idioms
    ]
    with open(idionly2vec_kv_path, 'w') as fh:
        # the first line is vocab_size dim_size
        # e.g. 1999995 300
        fh.write(" ".join([str(len(idioms)), str(args.vector_size)]) + "\n")
        for idiom, idiom_vec in zip(idioms, idiom_vectors):
            fh.write(idiom + " ")
            for comp in idiom_vec:
                fh.write(str(comp) + " ")  # write each component
            fh.write("\n")   # end with a new line.


if __name__ == '__main__':
    main()
