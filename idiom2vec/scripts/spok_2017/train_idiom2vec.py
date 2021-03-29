"""
trains an idiom2vec model.
"""
from gensim.models import Word2Vec
import logging
from sys import stdout
from datetime import datetime
from gensim.models.callbacks import CallbackAny2Vec
from idiom2vec.configs import IDIOM2VEC_DIR
from idiom2vec.corpora import Spok2017
from matplotlib import pyplot as plt

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


# parameters for training.
PARAMS = {
    'vector_size': 100,
    'window': 10,
    'min_count': 1,
    'workers': 5,
    'sg': 1,  # use skipgram
    'epochs': 50,  # number of iterations.
    'compute_loss': True  # want to have a look at the loss
}


def now_str() -> str:
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string


def main():
    global PARAMS
    # the corpus that are streamed.
    corpus = Spok2017()
    # instantiate the model.
    model = Word2Vec(sentences=corpus, **PARAMS, callbacks=[Idiom2VecCallback()])
    # save the model, after training it.
    save_path = str(IDIOM2VEC_DIR.joinpath("{}.model".format(now_str())))
    model.save(save_path)


if __name__ == '__main__':
    main()
