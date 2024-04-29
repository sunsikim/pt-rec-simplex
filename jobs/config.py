import logging

logging.basicConfig(
    format="%(asctime)s (%(module)s) (%(funcName)s) : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Files
LOCAL_DIR = "/tmp/simplex"
DATA_DIR = f"{LOCAL_DIR}/data"
RATINGS_FILE_NAME = "ratings.parquet"
MOVIE_DATA_FILE_NAME = "movies.parquet"
MOVIE_INDEX_MAP_FILE_NAME = "mid2idx.parquet"
ML_1M_SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_20M_SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
CHECKPOINT_FILE_NAME = "ckpt.pt"
TRAIN_LOG_FILE = "train_log.csv"
# Preprocess
HISTORY_MAX_LEN = 100
MIN_INTERACTIONS = 20
NEGATIVE_SAMPLE_POOL_SIZE = 1000
# Train
VALIDATION_USER_RATIO = 0.1
EMBEDDING_DIM = 200
LOSS_NEGATIVE_MARGIN = 0.9
LOSS_NEGATIVE_WEIGHT = 300
NEGATIVE_SAMPLE_SIZE = 100
BATCH_SIZE = 256
MAX_EPOCH = 100


class ExecutableJobs:

    def __init__(self, job_name: str = None):
        self.name = job_name

    def execute(self):
        raise NotImplementedError
