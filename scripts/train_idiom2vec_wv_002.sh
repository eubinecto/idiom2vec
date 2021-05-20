# idiom2vec 002
# trained with a coca_spok (full), and opensub (full)
# trained with raw sentences with simple clean up. (no stopwords filtering, no proper nouns filtering)
# the parameters are as follows
# They are all lemmatised. That's for sure.
python3 ../idiom2vec/runners/train_idiom2vec.py \
    --vector_size=200 \
    --window=8 \
    --min_count=1 \
    --workers=8 \
    --alpha=0.025 \
    --sg=1 \
    --epochs= 80 \
    --compute_loss \
    --model_version="002"\
    --train_with="word2vec"
