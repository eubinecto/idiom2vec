
import argparse
from gensim.models import Word2Vec
from idiom2vec.paths import IDIOM2VEC_WV_002_BIN
from identify_idioms.service import load_idioms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idiom", type=str,
                        default="catch-22")
    parser.add_argument("--top_n", type=str,
                        default=10)
    args = parser.parse_args()
    idiom: str = args.idiom
    top_n: int = args.top_n

    # print out the cosine similarity. - should mention that in the related work section.
    idiom2vec_002 = Word2Vec.load(IDIOM2VEC_WV_002_BIN)
    scores = idiom2vec_002.wv.most_similar(idiom, topn=None).tolist()
    sims = [
        (idiom2vec_002.wv.index_to_key[idx], score)
        for idx, score in enumerate(scores)
    ]
    # and then ... filter only the idioms
    idiom_keys = [
        idiom.replace(" ", "_")
        for idiom in load_idioms()
    ]

    idiom_sims = [
        (word, score)
        for word, score in sims
        if word in idiom_keys
    ]
    # sort it
    idiom_sims = sorted(idiom_sims, key=lambda x: x[1], reverse=True)
    # take the top_n only
    idiom_sims = idiom_sims[:top_n]

    print("The idioms that are similar to {}:".format(idiom))
    for idiom, score in idiom_sims:
        print(idiom, score)


if __name__ == '__main__':
    main()
