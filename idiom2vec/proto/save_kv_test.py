from gensim.models import Word2Vec, KeyedVectors
from identify_idioms.service import load_idioms


def main():
    idiom2vec_model = Word2Vec.load("../../data/idiom2vec/idiom2vec_001.model")
    # --- save idionly2vec --- #
    vector_size = 100
    idionly2vec_kv_path = "../../data/idiom2vec/idionly2vec_001.kv"
    idioms = [
        idiom
        for idiom in load_idioms()  # get this from identify idioms lib.
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
