# idiom2vec 002
# trained with a coca_spok (full), and opensub (full)
# trained with raw sentences. (no stopwords filtering)
# the parameters are as follows
python3 ../idiom2vec/runners/train_idiom2vec.py \
    --vector_size=200 \
    --window=8 \
    --min_count=1 \
    --workers=4\
    --sg=1 \
    --epochs=70 \
    --compute_loss \
    --model_version="002"\
    --train_with="word2vec"
