
from playground.gensim_pg.nearest_neighbours import load_fasttext_model

FASTTEXT_MODEL = load_fasttext_model()


def analogize(A: str, B: str, C: str):
    """
    this will return X, where:
    A - B = C - X. i.e. vector_offset(A, B) = vector_offset(C, X).
    that is, X = B - A + C
    """
    global FASTTEXT_MODEL
    A_vec = FASTTEXT_MODEL.get_vector(A)
    B_vec = FASTTEXT_MODEL.get_vector(B)
    C_vec = FASTTEXT_MODEL.get_vector(C)
    # why does the "ratio equation" not work?
    # A: Because they are vectors! and what you need is vector arithmetics.
    x = B_vec - A_vec + C_vec
    for sim in FASTTEXT_MODEL.similar_by_vector(x, topn=20):
        print(sim)
    print("########")


def main():
    global FASTTEXT_MODEL
    analogize("Korea", "Seoul", "UK")
    analogize("Korea", "Seoul", "Apple")  # what if A and C are not of the same entity?
    analogize("Apple", "iPhone", "Samsung")
    analogize("chemistry", "reaction", "mathematics")  # this one is so interesting.
    # now this one is one to which you may not have a clear answer
    analogize("engineer", "implements", "leader")  # you can learn from the AI.
    analogize("better", "best", "easier") # syntactic relationship


if __name__ == '__main__':
    main()
