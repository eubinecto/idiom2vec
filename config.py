

from pathlib import Path
from os import path
# the root directory of this project
# define the directories here
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(ROOT_DIR, "data")
# data directories
WORD2VEC_DIR = path.join(DATA_DIR, "word2vec")
OPENSUB_DIR = path.join(DATA_DIR, "opensub")
SLIDE_DIR = path.join(DATA_DIR, "slide")
# etc
PLAYGROUND_DIR = path.join(DATA_DIR, "playground")
GENSIM_PG_DIR = path.join(PLAYGROUND_DIR, "gensim_pg")
# the dataset to use and test
FASTTEXT_VEC_PATH = path.join(WORD2VEC_DIR, "crawl-300d-2M.vec")
FASTTEXT_PKL_PATH = path.join(WORD2VEC_DIR, "crawl-300d-2M.pkl")  # path to store the pickle binary
# delimiter to use for reading & writing data
DELIM = "\t"

# matcher related
IDIOM_MATCHER_PKL_PATH = path.join(SLIDE_DIR, 'idiom_matcher.pkl')
IDIOM_MATCHER_INFO_TSV_PATH = path.join(SLIDE_DIR, "idiom_matcher_info.tsv")

# spacy
NLP_MODEL = "en_core_web_sm"

# slide data
SLIDE_TSV_PATH = path.join(SLIDE_DIR, "slide.tsv")
SLIDE_POS_TSV_PATH = path.join(SLIDE_DIR, "slide_pos.tsv")

# corpus
CORPUS_DIR = path.join(DATA_DIR, "corpus")
SAMPLE_1_NDJSON_PATH = path.join(OPENSUB_DIR, "en_0_5297760_6728603.ndjson")
SAMPLE_2_NDJSON_PATH = path.join(OPENSUB_DIR, "en_0_2363046_4933283.ndjson")
SAMPLE_3_NDJSON_PATH = path.join(OPENSUB_DIR, "en_0_5614038_6583095.ndjson")
NDJSON_SAMPLES_PATH = [
    SAMPLE_1_NDJSON_PATH,
    SAMPLE_2_NDJSON_PATH,
    SAMPLE_3_NDJSON_PATH
]

# corpus.
DICTIONARY_PATH = path.join(CORPUS_DIR, "dictionary.corpus")
