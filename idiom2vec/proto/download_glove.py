import gensim.downloader as api


def main():
    glove = api.load("glove-wiki-gigaword-200")


if __name__ == '__main__':
    main()
