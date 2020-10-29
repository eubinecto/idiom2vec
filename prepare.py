
import numpy as np
# to be used for pre processing data
from functional import seq
from config import C2DEF_CSV_PATH
import pandas as pd


class Prepare:
    """
    1. load data
    2. clean texts
    3. tokenise texts
    4. build vocabulary
    5. build BOW wtih word2vec from vocab
    """
    @classmethod
    def load_c2def_csv(cls) -> pd.DataFrame:
        return pd.read_csv(C2DEF_CSV_PATH)

    @classmethod
    def exec(cls) -> np.ndarray:
        c2def_df = cls.load_c2def_csv()
        pass
