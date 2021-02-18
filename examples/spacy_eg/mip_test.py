from merge_idioms.builders import MIPBuilder


def main():
    sent = "That's a catch-22!"
    mip_builder = MIPBuilder()
    mip_builder.construct()
    mip = mip_builder.mip
    # use this pipeline!
    # now, what to do tomorrow -> build a effing corpus!
    for token in mip(sent):
        print(token.lemma_)


if __name__ == '__main__':
    main()
