from merge_idioms.builders import MIPBuilder
from termcolor import colored

# sentences to test
SENTENCES = (
    "What a lovely vase! You shouldn't have!",
    "Kate is willing to pay full price for an expensive handbag, but I just can't wrap my head around that.",
    "She wasn't that interested at first, but she loved it once she got the bit between her teeth.",
    # hyphenated
    "The bank's overdraft policy is a catch-22 for those trying to get out of poverty,"
    " as it charges you higher fees for having less money in your account.",
    "Established families tend to hold themselves above the johnny-come-lately.",
    # cased
    "Established families tend to hold themselves above the Johnny-come-lately.",
    # not hyphenated
    "The bank's overdraft policy is a catch 22 for those trying to get out of poverty,"
    " as it charges you higher fees for having less money in your account.",
    "Established families tend to hold themselves above the johnny come lately."
)


def main():
    mip_builder = MIPBuilder()
    mip_builder.construct()
    mip = mip_builder.mip
    for sent in SENTENCES:
        doc = mip(sent)
        lemmas = [token.lemma_ for token in doc]
        poses = [token.pos_ for token in doc]
        # don't use lex_ids. THat's not the one.
        lemma_id = [token.lemma for token in doc]
        tags = [token.tag_ for token in doc]
        for lemma, poses, lemma_id, tag in zip(lemmas, poses, lemma_id, tags):
            if tag == "IDIOM":
                colored_lemma = colored(lemma, 'magenta')
            else:
                colored_lemma = colored(lemma, 'blue')
            print("({}, {}, {}, {}),".format(colored_lemma, poses, str(lemma_id), tag), end=" ")
        print("\n##########")


if __name__ == '__main__':
    main()
