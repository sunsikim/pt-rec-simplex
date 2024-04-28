import logging

logging.basicConfig(
    format="%(asctime)s (%(module)s) (%(funcName)s) : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

LOCAL_DIR = "/tmp/simplex"
DATA_DIR = f"{LOCAL_DIR}/data"
ML_1M_SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_20M_SOURCE_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
HISTORY_MAX_LEN = 100


class ExecutableJobs:

    def __init__(self, job_name: str = None):
        self.name = job_name

    def execute(self):
        raise NotImplementedError
