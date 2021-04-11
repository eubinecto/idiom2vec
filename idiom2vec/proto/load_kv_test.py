from gensim.models import KeyedVectors


def main():
    idionly2vec_kv_path = "../../data/idiom2vec/idionly2vec_001.kv"
    idionly2vec_kv = KeyedVectors.load_word2vec_format(idionly2vec_kv_path, binary=False)

    print("# --- idioms similar to american dream --- #")
    for key, sim in idionly2vec_kv.most_similar('american_dream'):
        print(key, sim)


if __name__ == '__main__':
    main()
