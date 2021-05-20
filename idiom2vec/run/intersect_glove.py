"""
Intersect glove into idiom2vec.
Not exactly the same as transfer learning.
(But why was in shown to have no significant benefits?)
"""
import argparse
from gensim.models import Word2Vec, KeyedVectors

from idiom2vec.paths import IDIOM2VEC_WV_001_BIN, IDIOM2VEC_WV_002_BIN, IDIOM2VEC_WV_003_BIN, GLOVE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_version',
                        type=str,
                        default="002")
    args = parser.parse_args()
    # --- paths setup --- #
    if args.model_version == "001":
        idiom2vec_bin_path = IDIOM2VEC_WV_001_BIN
    elif args.model_version == "002":
        idiom2vec_bin_path = IDIOM2VEC_WV_002_BIN
    elif args.model_version == "003":
        idiom2vec_bin_path = IDIOM2VEC_WV_003_BIN
    else:
        raise ValueError

    idiom2vec = Word2Vec.load(idiom2vec_bin_path)
    idiom2vec.wv.intersect_word2vec_format(GLOVE, binary=True)


if __name__ == '__main__':
    main()
