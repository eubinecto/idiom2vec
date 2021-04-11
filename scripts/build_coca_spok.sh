# the dirs and paths to be used.
CORPORA_DIR="../data/coca_spok/corpora"
ORIGIN_TXT_PATH="../data/coca_spok/origin.txt"
ORIGIN_SPLITS_DIR="../data/coca_spok/origin_splits"
TRAIN_SPLITS_DIR="../data/coca_spok/train_splits"
TRAIN_SPLITS_FS_TSV_PATH="../data/coca_spok/train_splits/fs_manifest.tsv"
TRAIN_NDJSON_PATH="../data/coca_spok/train.ndjson"

python3 ../idiom2vec/runners/build_origin.py \
  --corpora_dir=$CORPORA_DIR \
  --origin_txt_path=$ORIGIN_TXT_PATH
## split the original corpus into small files
python3 ../idiom2vec/runners/build_origin_splits.py \
  --origin_txt_path=$ORIGIN_TXT_PATH \
  --split_size=20000 \
  --origin_splits_dir=$ORIGIN_SPLITS_DIR
## tokenise the splits
python3 ../idiom2vec/runners/build_train_splits.py \
  --origin_splits_dir=$ORIGIN_SPLITS_DIR \
  --train_splits_dir=$TRAIN_SPLITS_DIR \
  --train_splits_fs_path=$TRAIN_SPLITS_FS_TSV_PATH
# merge the tokens into a train-ready file
python3 ../idiom2vec/runners/merge_train_splits.py \
  --train_splits_dir=$TRAIN_SPLITS_DIR \
  --train_ndjson_path=$TRAIN_NDJSON_PATH