from typing import Optional

import numpy as np


class Context2Def:

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        starting with an unsupervised way. just memorize it
        :param X_train: (num_def, embedding_dim). the bag of words rep. of the definitions
        :param y_train: (num_def,). the labels. definitions in string.
        """
        self.X_train: np.ndarray = X_train
        self.y_train: np.ndarray = y_train
        self.__dists:  Optional[np.ndarray] = None

    def fit(self, X_test: np.ndarray) -> np.ndarray:
        """
        :param X_test: (num_context, embedding_dim)
        :return: (num_context,)
        """
        self.__dists = self._comp_dists(X_test)  # (num_context, embedding_dim) -> (num_context, num_defs)
        nn_indices = np.argsort(self.__dists, axis=1)[:, 0]  # (num_context, num_defs) -> (num_context,)
        nn_labels = self.y_train[nn_indices]  # (num_def,) -> (num_context,)
        return nn_labels

    def _comp_dists(self, X_test: np.ndarray, metric='L1') -> np.ndarray:
        pass
