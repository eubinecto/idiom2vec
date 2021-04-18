# idiom2vec 001
# the parameters are as follows
python3 ../idiom2vec/runners/train_idiom2vec.py \
    --vector_size=200 \
    --window=8 \
    --min_count=1 \
    --workers=4\
    --epochs=30 \
    --compute_loss \
    --dm_concat=0 \
    --dbow_words=1 \
    --model_version="001"\
    --train_with="doc2vec"