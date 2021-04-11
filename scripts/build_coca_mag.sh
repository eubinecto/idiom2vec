# the dirs and paths to be used.
CORPORA_DIR="../data/coca_mag/corpora"
ORIGIN_TXT_PATH="../data/coca_mag/origin.txt"
ORIGIN_SPLITS_DIR="../data/coca_mag/origin_splits"
TRAIN_SPLITS_DIR="../data/coca_mag/train_splits"
TRAIN_SPLITS_FS_TSV_PATH="../data/coca_mag/train_splits/fs_manifest.tsv"
TRAIN_NDJSON_PATH="../data/coca_mag/train.ndjson"

# first, build origin.txt.
echo "running build_origin.py..."
python3 ../idiom2vec/runners/build_origin.py \
  --corpora_dir=$CORPORA_DIR \
  --origin_txt_path=$ORIGIN_TXT_PATH
## split the original corpus into small files
echo "running build_origin_splits.py..."
python3 ../idiom2vec/runners/build_origin_splits.py \
  --origin_txt_path=$ORIGIN_TXT_PATH \
  --split_size=20000 \
  --origin_splits_dir=$ORIGIN_SPLITS_DIR
## tokenise the splits
echo "running build_train_splits.py..."
python3 ../idiom2vec/runners/build_train_splits.py \
  --num_workers=12 \
  --corpus_name="coca_mag" \
  --origin_splits_dir=$ORIGIN_SPLITS_DIR \
  --train_splits_dir=$TRAIN_SPLITS_DIR \
  --train_splits_fs_path=$TRAIN_SPLITS_FS_TSV_PATH
# merge the tokens into a train-ready file
echo "running merge_train_splits.py..."
python3 ../idiom2vec/runners/merge_train_splits.py \
  --train_splits_dir=$TRAIN_SPLITS_DIR \
  --train_ndjson_path=$TRAIN_NDJSON_PATH