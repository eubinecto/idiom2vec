## 28th of March 2021

> First, understand the format of COCA spoken text

1. each line starts with a label for a certain source.
  - e.g. @@4172426 
  - e.g. @@4172430 
2. each utterance is conveniently tagged with the speaker of the utterance.
  - e.g. @!TERRY-GROSS# Andy Greenberg is ...
  - e.g. @!DAVE-DAVIES# So let 's talk about what 's in 
3. profanities are censored with `@ @ @ @ @ @ @ @ @ @` (always 10 of them).
  - e.g. The individual mandate was meant @ @ @ @ @ @ @ @ @ @ for making that decision.
  - e.g. , and @ @ @ @ @ @ @ @ @ @ . Or , you know , that typically means higher deductibles and higher co-payments.
4. Punctuations are delimited with space.
  - e.g. ealth insurance . The subsidies u
  - e.g. Obamacare enrollees ? Could they af
  - e.g. one who is 64 , right no


> Can you exploit any of them?

1. the labels
 - Maybe useful if I want to retrieve examples and want to show the source. But as far as idiom2vec project
 - this comes in handy for parallel processing! you know exactly how to split the corpus up.
is concerned, I don't need them.
2. speaker segmentation.
 - might be useful for chatbot. But not really for me. I'll just use sliding windows.
3. profanities
 - could be a useful information. encode that with `[profanity]`. 
4. Punctuations. 
 - May not really be useful, I think. Ignore this part. 


## 29th of March 2021

> How do I tune hyperparameters for Word2Vec?

How many epochs should I do? Would visualising the loss help?


I had set the epoch to 10, and this was the result:
![](.journal_images/baf42800.png)

hyper parameters for the graph above:
```python
PARAMS = {
    'vector_size': 100,
    'window': 10,
    'min_count': 1,
    'workers': 5,
    'sg': 1,  # use skipgram
    'epochs': 10,  # number of iterations.
    'compute_loss': True  # want to have a look at the loss
}
```


You should either increase the learning rate, or... 
increase the epoch size.

Let's try 50 epochs.
