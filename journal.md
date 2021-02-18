17h of December

--- 
## what to do?
- [ ] just try to train word2vec with the data you have. 
- [ ] see the result, and reflect on what you should improve on. and what's next.


## The problem with the big one
It has too many duplicates. 

Does it matter when training word2vec? Could it even help? (I mean,it's 
kind of like over-sampling with copy & paste).

If there were repeated sentences in the dataset, would it impact what word2vec learns?

I.e. would those duplicated sentence be more weighted than the others?

What if the number of duplicated sentences were exactly the same for each sentence? 
Is that the case for the big one?

## I've got an error while tokenzing the data
after running toknize_subs.py:
```
/Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vecenv/bin/python /Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vec/opensub/scripts/tokenize_subs.py
Traceback (most recent call last):
  File "/Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vec/opensub/scripts/tokenize_subs.py", line 38, in <module>
    main()
  File "/Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vec/opensub/scripts/tokenize_subs.py", line 30, in main
    tokens = tokenize_sub(sub)
  File "/Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vec/opensub/scripts/tokenize_subs.py", line 15, in tokenize_sub
    doc = idiom_nlp(sub)
  File "/Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vec/slide/utils.py", line 118, in __call__
    return self.nlp(text.strip().lower())
  File "/Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vecenv/lib/python3.8/site-packages/spacy/language.py", line 984, in __call__
    doc = proc(doc, **component_cfg.get(name, {}))
  File "/Users/eubin/Desktop/Projects/Big/idiom2vec/idiom2vec/slide/utils.py", line 86, in __call__
    retokeniser.merge(doc[start:end],
  File "spacy/tokens/_retokenize.pyx", line 55, in spacy.tokens._retokenize.Retokenizer.merge
ValueError: [E199] Unable to merge 0-length span at `doc[4:4]`.
```
for which sentence the error occurs?
```
["die"]
["stuart", "sterling", "ancestor"]
["fight", "battle", "peru", "see", "atahualpa", "dead", "marry", "inca", "princess"]

```
```
...
He died in...
Stuart Sterling's ancestor was one of them,
He fought in the battles in Peru, saw Atahualpa dead, married an Inca princess,
But he had blood on his hands, <- this sentence
```

and we have this idiom:
```tsv
5930902300252675198	have blood on one's hands	"[[{""LEMMA"": ""have""}, {""LEMMA"": ""blood""}, {""LEMMA"": ""on""}, {""TAG"": ""PRP$""}, {""LEMMA"": ""hand""}]]"
```

Oh.. I see the problem here. It's because it has already been merged..
```tsv
8246625119345375174	on one's hands	"[[{""LEMMA"": ""on""}, {""TAG"": ""PRP$""}, {""LEMMA"": ""hand""}]]"
```
you have this, and that. How would you deal with this case? well, let's have a think about this later. for now, 
just skip the exception.

```
["create", "art", "example"]
[]
["thank"]
["lady", "sort", "pepper", "pepper", "great", "local", "speciality"]
```
ah.. didn't think of the case where the tokens after filtering out are empty.


also, I'm passing quite a lot of idioms. It is still running, but here are a few exmaples of that
```
pass merging for:on one's hands
pass merging for:go for it
done:10001
done:20001
pass merging for:line in the sand
done:30001
done:40001
pass merging for:end of the world
pass merging for:end of the world
pass merging for:on the table
done:50001

```
-> I've fixed that. That took quite a time.

## SLA reading - notes
Input's and outputs. That's what I want.


## Some reflection
hey, you really should speed things up here. 

For some, this kind of training may work. But for some, it won't, because of some domain-specific properties

1. they are conversations, and conversations are short. (e.g. It's a catch-22). So setting the window "around" the word
may not be ideal to have the model learn the semantics of the word. What might work best is to set the window "before"
the target word. -> If we are doing it this way, I might have to implement the model by myself. (preferably using pytorch)


## reflection - idiom2vec + topic modeling
What I want to do are two things.
