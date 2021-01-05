from pathlib import Path
from os import path
from datetime import datetime
# the root directory of this project
# define the directories here
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = path.join(ROOT_DIR, "data")


# to log everything. I'll put date and time to everything.
def now() -> str:
    now_obj = datetime.now()
    return now_obj.strftime("%d_%m_%Y__%H_%M_%S")


# to be used by other models
now_str = now()

# data directories
WORD2VEC_DIR = path.join(DATA_DIR, "word2vec")
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

# opensub
OPENSUB_DIR = path.join(DATA_DIR, "opensub")
SAMPLE_1_NDJSON_PATH = path.join(OPENSUB_DIR, "en_0_5297760_6728603.ndjson")
SAMPLE_2_NDJSON_PATH = path.join(OPENSUB_DIR, "en_0_2363046_4933283.ndjson")
SAMPLE_3_NDJSON_PATH = path.join(OPENSUB_DIR, "en_0_5614038_6583095.ndjson")
NDJSON_SAMPLES_PATH = [
    SAMPLE_1_NDJSON_PATH,
    SAMPLE_2_NDJSON_PATH,
    SAMPLE_3_NDJSON_PATH
]
SUBS_PATH = path.join(OPENSUB_DIR, "OpenSubtitles.en-pt_br_en.txt")  # the conversation dataset
SUBS_TOKENIZED_NDJSON_PATH = path.join(OPENSUB_DIR, "OpenSubtitles.en-pt_br_en_tokenized.ndjson")

# corpus
CORPUS_DIR = path.join(DATA_DIR, "corpus")
DICTIONARY_PATH = path.join(CORPUS_DIR, "dictionary.corpus")


# idiom2vec
IDIOM2VEC_DIR = path.join(DATA_DIR, 'idiom2vec')
IDIOM2VEC_MODELS_DIR = path.join(IDIOM2VEC_DIR, "models")
IDIOM2VEC_LOGS_DIR = path.join(IDIOM2VEC_DIR, "logs")
IDIOM2VEC_PKL_PATH = path.join(IDIOM2VEC_MODELS_DIR, "idiom2vec{}.pkl".format(now_str))
