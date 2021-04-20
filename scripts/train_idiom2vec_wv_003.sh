# idiom2vec 003
# trained with a coca_spok (full), and opensub (full)
# trained with raw sentences. (no stopwords filtering)
# starting with projection weights of glove (transfer learning)
# the parameters are as follows
python3 ../idiom2vec/runners/train_idiom2vec.py \
    --vector_size=200 \
    --window=8 \
    --min_count=1 \
    --workers=8 \
    --alpha=0.025 \
    --sg=1 \
    --epochs=85 \
    --compute_loss \
    --remove_stopwords \
    --remove_propns \
    --model_version="003"\
    --train_with="word2vec"
