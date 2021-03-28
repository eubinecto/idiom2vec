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
is concerend, I don't need them.
2. speaker segmentation.
 - might be useful for chatbot. But not really for me. Just set the window.
3. profanities
 - could be a useful information. encode that with `[profanity]`. 
4. Puncutations. 
 - May not really be useful, I think. Ignore this part. 
