# named entity recognition with spacy

import spacy

sentences = u"""Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops and sells consumer electronics, computer software, and online services.
 It is considered one of the Big Tech technology companies, alongside Amazon, Google, Microsoft, and Facebook."""\
.split("\n")
# use this language model
nlp = spacy.load(name='en_core_web_sm')


"""
spacy has the following entites:
PERSON: People, including fictional ones
NORP: Nationalities or religious or political groups
FACILITY: Buildings, airports, highways, bridges, and so on
ORG: Companies, agencies, institutions, and so on
GPE: Countries, cities, and states
LOC: Non GPE locations, mountain ranges, and bodies of water
PRODUCT: Objects, vehicles, foods, and so on (not services)
EVENT: Named hurricanes, battles, wars, sports events, and so on
WORK_OF_ART: Titles of books, songs, and so on
LAW: Named documents made into laws
LANGUAGE: Any named language
"""


def main():
    global sentences, nlp
    # generate doc objects with spacy
    docs = [nlp(sentence) for sentence in sentences]
    for doc in docs:
        if not doc.ents:
            print("no entities detected")
        else:
            for ent in doc.ents:
                print(ent.text, ent.label_)
            print("----")


if __name__ == '__main__':
    main()
