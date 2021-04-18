from idiom2vec.corpora import IdiomSentences


def main():
    sents = IdiomSentences()

    for sent in sents:
        print(sent)


if __name__ == '__main__':
    main()
