"""
trains an idiom2vec model.
"""
from gensim.models import Word2Vec
import logging
from sys import stdout
from datetime import datetime
from gensim.models.callbacks import CallbackAny2Vec
from idiom2vec.configs import IDIOM2VEC_DIR
from idiom2vec.corpora import CocaSpok, CocaFict
from matplotlib import pyplot as plt
import argparse
logging.basicConfig(stream=stdout, level=logging.INFO)


# to be used for printing out the loss
class Idiom2VecCallback(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 0
        self.losses = list()  # collect losses here

    def on_epoch_end(self, model):
        """
        this will be called at the end of each epoch.
        """
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1
        self.losses.append(loss)
        self.visualise_loss()  # plot the loss

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
                        default=100)
    # the size of the window
    parser.add_argument('--window',
                        type=int,
                        default=10)
    # only the words with a frequency above the minimum count will be included in the vocab.
    parser.add_argument('--min_count',
                        type=int,
                        default=1)
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
                        default=50)
    # compute loss
    parser.add_argument('--compute_loss',
                        dest='compute_loss',
                        default=False,
                        action='store_true')

    # the paths
    parser.add_argument('--log_path',
                        type=str,
                        default=...)
    parser.add_argument('--idiom2vec_model_path',
                        type=str,
                        default=...)
    parser.add_argument('--idiom2vec_light_kv_path',
                        type=str,
                        default=...)
    args = parser.parse_args()

    # --- params setup --- #
    # the parameters to pass to word2vec.
    params = {
        'vector_size': args.vector_size,
        'window': args.window_size,
        'min_count': args.min_count,
        'workers': args.workers,
        'sg': args.sg,
        'epochs': args.epochs,
        'compute_loss': args.compute_loss
    }

    # --- logger setup --- #
    logger = logging.getLogger("train_idiom2vec")

    # --- prepare the corpus --- #
    # the corpus that are streamed.
    coca_spok = CocaSpok()
    coca_fict = CocaFict()
    # TODO: chain the corpus?
    corpus = ...  # chain corpus.

    # --- train the idiom2vec_model
    # instantiate the idiom2vec_model.
    idiom2vec_model = Word2Vec(sentences=corpus,
                               **params,
                               callbacks=[Idiom2VecCallback()])
    # save the idiom2vec_model, after training it.
    idiom2vec_model.save(args.idiom2vec_model_path)

    # TODO: then, save idiom2vec_light.kv.



if __name__ == '__main__':
    main()
