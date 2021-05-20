# idiom2vec 003
# trained with a coca_spok (full), and opensub (full)
# stopwords are removed. propns are removed.
# the parameters are as follows
python3 ../idiom2vec/runners/train_idiom2vec.py \
    --vector_size=200 \
    --window=8 \
    --min_count=1 \
    --workers=4 \
    --alpha=0.025 \
    --sg=1 \
    --epochs=130 \
    --compute_loss \
    --remove_stopwords \
    --remove_propns \
    --intersect_glove \
    --model_version="004"\
    --train_with="word2vec"
