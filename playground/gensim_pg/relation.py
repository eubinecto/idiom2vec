
from playground.gensim_pg.nearest_neighbours import load_fasttext_model

FASTTEXT_MODEL = load_fasttext_model()


def find_analogy(A: str, B: str, C: str):
    """
    this will return X, where:
    A:B :: C: X
    # Q: but why only add or subtract
    X = B - A + C
    """
    global FASTTEXT_MODEL
    A_vec = FASTTEXT_MODEL.get_vector(A)
    B_vec = FASTTEXT_MODEL.get_vector(B)
    C_vec = FASTTEXT_MODEL.get_vector(C)
    x = B_vec - A_vec + C_vec  # but why does this work? # why does solving the ratio equation not work?
    for sim in FASTTEXT_MODEL.similar_by_vector(x):
        print(sim)
    print("########")


def main():
    global FASTTEXT_MODEL
    # Iphone: Apple = x : Samsung
    # x = Iphone * Samsung / Apple
    find_analogy("Korea", "Seoul", "UK")
    find_analogy("Apple", "iPhone", "Samsung")
    find_analogy("Korea", "Seoul", "Apple")  # what if A and C are not of the same entity?


if __name__ == '__main__':
    main()
