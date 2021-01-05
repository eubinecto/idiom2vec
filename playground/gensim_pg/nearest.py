from playground.gensim_pg.nearest_neighbours import load_fasttext_model
from termcolor import colored

# 게으른 완벽주의자 보다는
# 성실한 불완전함이 더 좋다.
# 이것에 대해서는 다들 어떻게 생각하나?
# 불완전해도 일단 해내는 것! 그런 마음이 중요하다.


# load the model here!
fasttext_model = load_fasttext_model()


fasttext_model.

# helper function
def print_nn(word: str):
    global fasttext_model
    print(colored("#####similar words to {}#######".format(word), 'blue'))
    for sim in fasttext_model.similar_by_word(word, topn=25):
        print(sim)


def main():
    print_nn("cat")
    # is this frequently used?
    print_nn("backpropagation")
    # okay, how about this?
    print_nn("autoencoder")
    # konglish?
    print_nn("webtoon")
    # this is one of words in the list of "the most infrequently used words"
    print_nn("sobriquet")
    # 아.. 이 정도로 해보자. 어느정도 유사한 단어를 찾아주는 것은 꽤나 잘 해내는 것 같다.
    # 하지만 그렇다고 해서 이 모델의 cat이라는 단어에 대한 loss와 ahjumma라는 단어에 대한 loss가
    # 비슷할 것이라고 말할 수 있을까? 아님.


if __name__ == '__main__':
    main()
