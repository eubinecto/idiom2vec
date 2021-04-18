from gensim.models import Word2Vec
from identify_idioms.service import load_idioms
from idiom2vec.paths import IDIOM2VEC_WV_001_BIN, IDIONLY2VEC_001_KV


def main():
    idiom2vec_model = Word2Vec.load(IDIOM2VEC_WV_001_BIN)
    # --- save idionly2vec --- #
    vector_size = 200
    idionly2vec_kv_path = IDIONLY2VEC_001_KV

    keys = [
        idiom.replace(" ", "_")
        for idiom in load_idioms()
    ]
    idioms = [
        idiom
        for idiom in keys  # get this from identify idioms lib.
        if idiom2vec_model.wv.key_to_index.get(idiom, None)
    ]
    idiom_vectors = [
        idiom2vec_model.wv.get_vector(idiom)
        for idiom in idioms
    ]
    with open(idionly2vec_kv_path, 'w') as fh:
        # the first line is vocab_size dim_size
        # e.g. 1999995 300
        fh.write(" ".join([str(len(idioms)), str(vector_size)]) + "\n")
        for idiom, idiom_vec in zip(idioms, idiom_vectors):
            idiom = idiom.replace(" ", "_")  # this is necessary. if you don't, the model can't read anything.
            fh.write(idiom + " ")
            for comp in idiom_vec:
                fh.write(str(comp) + " ")  # write each component
            fh.write("\n")   # end with a new line.


if __name__ == '__main__':
    main()
