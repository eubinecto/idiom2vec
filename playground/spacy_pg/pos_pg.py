# part-of-speech tagging with spacy.
# use this to remove words with particular pos, for example.

import spacy

sentences = u"""How did this happen to me?
You made a deal, you stupid son of a bitch.
You made a deal with Malebolgia.
You cut a deal for your soul. Cut a deal.
The deal was you'd see Wanda and then become Hellspawn... a ranking officer in the devil's army.
Well, now you've seen her.
Time to pay the piper.""".split("\n")
# use this language model
nlp = spacy.load(name='en_core_web_sm')


def main():
    global sentences, nlp
    docs = [nlp(sentence) for sentence in sentences]

    # print out part of speech
    for doc in docs:
        sentence_pos_tagged = [
            (token.text, token.pos_)
            for token in doc
        ]
        # print out words in the sentence with their pos
        for word, pos in sentence_pos_tagged:
            print(word + " --> " + pos)
        print("----")


if __name__ == '__main__':
    main()
