# idiom2vec 001
# the parameters are as follows
python3 ../idiom2vec/runners/train_idiom2vec.py \
    --vector_size=200 \
    --window=8 \
    --min_count=1 \
    --workers=4\
    --sg=1 \
    --epochs=100 \
    --compute_loss \
    --model_version="001"\
    --train_with="word2vec"
