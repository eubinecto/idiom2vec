from playground.gensim_pg.nearest_neighbours import load_fasttext_model

model = load_fasttext_model()


def analogize(x: str, y: str, z: str):
    """
    x is to y, as z is to... what?
    """
    global model
    x_vec = model.get_vector(x)
    y_vec = model.get_vector(y)
    z_vec = model.get_vector(z)
    what_vec = z_vec + y_vec - x_vec
    print("-------")
    for word, dist in model.similar_by_vector(what_vec):
        print(word, ":", dist)


def main():
    analogize("engineer", "implements", "leader")


if __name__ == '__main__':
    main()
