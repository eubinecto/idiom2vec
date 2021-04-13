# idiom2vec 002
# corpus: coca_spok. doc_is_sent=False.
# the parameters are as follows
python3 ../idiom2vec/runners/train_idiom2vec.py \
    --vector_size=300 \
    --window=10 \
    --min_count=1 \
    --workers=46\
    --sg=1 \
    --epochs=50 \
    --compute_loss \
    --doc_is_sent \
    --idiom2vec_model_path="../data/idiom2vec/idiom2vec_002.model" \
    --idionly2vec_kv_path="../data/idiom2vec/idionly2vec_002.kv" \
    --coca_spok_train_ndjson_path="../data/coca_spok/train.ndjson"
