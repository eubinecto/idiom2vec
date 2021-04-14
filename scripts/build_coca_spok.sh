
echo "running build_origin.py..."
python3 ../idiom2vec/runners/build_origin.py \
  --corpus_name="coca_spok"
## split the original corpus into small files
echo "running build_origin_splits.py..."
python3 ../idiom2vec/runners/build_origin_splits.py \
  --split_size=200000 \
  --corpus_name="coca_spok"
## tokenise the splits
echo "running build_train_splits.py..."
python3 ../idiom2vec/runners/build_train_splits.py \
  --num_workers=6 \
  --corpus_name="coca_spok" \
# merge the tokens into a train-ready file
echo "running merge_train_splits.py..."
python3 ../idiom2vec/runners/merge_train_splits.py \
  --corpus_name="coca_spok"
