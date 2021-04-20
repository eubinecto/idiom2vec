
from idiom2vec.paths import GLOVE_6B_200D_TXT
def main():
    cont = 0
    with open(GLOVE_6B_200D_TXT, 'r') as fh:
        for line in fh:
            cont += 1

    print(cont)

if __name__ == '__main__':
    main()