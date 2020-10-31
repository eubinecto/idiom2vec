import pprint
from termcolor import colored
# from nltk import download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# for comparing similarities
from playground.gensim_pg import load_fasttext_model
import numpy as np

# testing with these
IDIOM = "fair and square"
SENSE_1 = "with absolute accuracy"
SENSE_2 = "honestly and straightforwardly"
# context for sense 1
CONTEXT_1 = "see it right there dude and doom you know" \
            " beat him fair and square raising powerful Anthony but I don't"
# context for sense 2
CONTEXT_2 = "just took their guy and just said we've voted him in " \
            "fair and square and you guys just removed him"


def main():
    """
    filtering out stop words with nltk
    credit: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    :return:
    """
    pp = pprint.PrettyPrinter(indent=4)
    # stop words
    print(colored("The list of stop words", 'blue'))
    # download it if you have not done it yet
    # download('stopwords')
    # have a look at the list of stop words provided by nltk
    en_stopwords = stopwords.words('english')
    # n't should be added
    en_stopwords.append('n\'t')
    en_stopwords.append('\'ve')
    pp.pprint(en_stopwords)

    # tokenizer
    print(colored("Tokenizing the context, the two senses and the idiom into words with nltk", 'blue'))
    # use nltk's word tokenizer
    idiom = word_tokenize(IDIOM.lower())
    sense_1 = word_tokenize(SENSE_1.lower())
    sense_2 = word_tokenize(SENSE_2.lower())
    context_1 = word_tokenize(CONTEXT_1.lower())
    context_2 = word_tokenize(CONTEXT_2.lower())
    print(IDIOM)
    print(SENSE_1)
    print(SENSE_2)
    print(CONTEXT_1)
    print("-they are tokenized to->")
    print(idiom)
    print(sense_1)
    print(sense_2)
    print(context_1)
    print(context_2)

    # filtering out stopwords
    print(colored("Stopwords should be removed", 'blue'))
    sense_1_no_sw = [token for token in sense_1 if token not in en_stopwords]
    sense_2_no_sw = [token for token in sense_2 if token not in en_stopwords]
    context_1_no_sw = [token for token in context_1 if token not in en_stopwords]
    context_2_no_sw = [token for token in context_2 if token not in en_stopwords]
    print(sense_1_no_sw)
    print(sense_2_no_sw)
    print(context_1_no_sw)
    print(context_2_no_sw)

    # filter out idioms
    print(colored("THe idiom in the context should also be removed (due to non-compositionality of idioms)", 'blue'))
    context_1_no_sw_sn = [token for token in context_1_no_sw if token not in idiom]
    context_2_no_sw_sn = [token for token in context_2_no_sw if token not in idiom]
    print(context_1_no_sw_sn)
    print(context_2_no_sw_sn)

    # compare sims
    print(colored("Generate similarities of the context to senses", 'blue'))
    fasttext_model = load_fasttext_model()
    fasttext_model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    dist_1 = fasttext_model.wmdistance(context_1_no_sw_sn, sense_1_no_sw)
    dist_2 = fasttext_model.wmdistance(context_1_no_sw_sn, sense_2_no_sw)
    print(dist_1, dist_2)
    dist_1 = fasttext_model.wmdistance(context_2_no_sw_sn, sense_1_no_sw)
    dist_2 = fasttext_model.wmdistance(context_2_no_sw_sn, sense_2_no_sw)
    print(dist_1, dist_2)


if __name__ == '__main__':
    main()


