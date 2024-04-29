import shutil
import requests
import logging
import random
import polars as pl
import jobs.config as config
import pathlib
from jobs.config import ExecutableJobs


class PreprocessJob:

    def __init__(self, job_name: str):
        self._data_type = f"ml-{job_name.split('-')[1]}"
        self._local_dir = pathlib.Path(config.LOCAL_DIR)
        self._movielens_dir = self._local_dir.joinpath(self._data_type)
        self._data_dir = pathlib.Path(config.DATA_DIR)

    def execute(self):
        self._local_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"download {self._data_type} data from source_url")
        self._download_data()

        logging.info("preprocess raw data into polars dataframe")
        movies = self._load_data("movies")
        ratings = self._load_data("ratings")
        num_users = ratings["user_id"].n_unique()
        num_items = ratings["movie_id"].n_unique()
        logging.info(f"number of unique users : {num_users:,} movies : {num_items:,}")

        logging.info("binarize and filter out some data to reduce sparsity")
        ratings = self._filter_data(ratings)
        logging.info(f"number of ratings left after filtering : {ratings.height:,}")

        logging.info("define user index map for filtered ratings data")
        uid2idx = ratings.select("user_id").unique().with_row_index("user_idx")
        mid2idx = ratings.select("movie_id").unique().with_row_index("movie_idx", 1)
        logging.info(f"unique users : {uid2idx.height:,} movies : {mid2idx.height:,}")

        logging.info("convert user_id, movie_id and reshape data")
        ratings = (
            ratings.join(uid2idx, on="user_id")
            .join(mid2idx, on="movie_id")
            .drop(["user_id", "movie_id"])
        )
        ratings = self._reshape_data(ratings)

        logging.info("save preprocessed dataset and index maps")
        self._data_dir.mkdir(exist_ok=True, parents=True)
        ratings.write_parquet(self._data_dir.joinpath(config.RATINGS_FILE_NAME))
        movies.write_parquet(self._data_dir.joinpath(config.MOVIE_DATA_FILE_NAME))
        mid2idx.write_parquet(self._data_dir.joinpath(config.MOVIE_INDEX_MAP_FILE_NAME))

    def _download_data(self):
        source = (
            config.ML_1M_SOURCE_URL
            if self._data_type == "ml-1m"
            else config.ML_20M_SOURCE_URL
        )
        with open(self._local_dir.joinpath(f"{self._data_type}.zip"), "wb") as file:
            response = requests.get(source)
            file.write(response.content)
        shutil.unpack_archive(
            filename=self._local_dir.joinpath(f"{self._data_type}.zip"),
            extract_dir=self._local_dir,
            format="zip",
        )

    def _load_data(self, file_prefix: str) -> pl.DataFrame:
        """
        File format of ML-1M and ML-20M are different, so implement in corresponding child class
        :param file_prefix: file name without extension
        :return: data loaded as polars dataframe
        """
        raise NotImplementedError

    def _filter_data(self, ratings: pl.DataFrame) -> pl.DataFrame:
        """
        1. Filter out data whose rating is less than 3.5 to binarize data
        2. Filter out users/movies with less interaction to reduce sparsity as in https://arxiv.org/pdf/2006.15516.pdf
        Rule: exclude user/item with number of interactions is less than self._min_interactions
        """
        filtered_ratings = ratings.filter(pl.col("rating") >= 3.5).drop(
            ["rating", "timestamp"]
        )
        valid_movies = (
            filtered_ratings.group_by("movie_id")
            .len(name="num_interactions")
            .filter(pl.col("num_interactions") >= config.MIN_INTERACTIONS)
            .drop("num_interactions")
        )
        filtered_ratings = filtered_ratings.join(
            valid_movies, on="movie_id", how="inner"
        )
        valid_users = (
            filtered_ratings.group_by("user_id")
            .len(name="num_interactions")
            .filter(
                (pl.col("num_interactions") >= config.MIN_INTERACTIONS).and_(
                    pl.col("num_interactions") <= config.HISTORY_MAX_LEN
                )
            )
            .drop("num_interactions")
        )
        return filtered_ratings.join(valid_users, on="user_id", how="inner")

    def _reshape_data(self, ratings: pl.DataFrame) -> pl.DataFrame:
        unique_items = set(ratings["movie_idx"])
        aggregated_data = (
            ratings.sample(fraction=1, shuffle=True)
            .group_by("user_idx")
            .agg(pl.col("movie_idx").alias("interacted_items"))
        )
        parsed_data = []
        for user_data in aggregated_data.to_dicts():
            history = user_data.pop("interacted_items")
            negative_items = unique_items.difference(history)
            negative_sample_pool = random.choices(
                population=list(negative_items), k=config.NEGATIVE_SAMPLE_POOL_SIZE
            )
            parsed_data.append(
                {
                    "user_idx": user_data["user_idx"],
                    "interacted_items": history,
                    "negative_sample_pool": negative_sample_pool,
                }
            )
        return pl.DataFrame(parsed_data)


class Preprocess1MJob(PreprocessJob, ExecutableJobs):
    """
    Download MovieLens 1M data from source, preprocess raw data and save preprocessed data
    """

    def __init__(self):
        job_name = "preprocess-1m"
        PreprocessJob.__init__(self, job_name)
        ExecutableJobs.__init__(self, job_name)

    def _load_data(self, file_prefix: str) -> pl.DataFrame:
        """
        Within context manager, special encoding is selected to avoid UnicodeDecodeError
        """
        if file_prefix == "movies":
            schema = {
                "movie_id": pl.Int32,
                "title": pl.String,
                "genres": pl.String,
            }
        elif file_prefix == "ratings":
            schema = {
                "user_id": pl.Int32,
                "movie_id": pl.Int32,
                "rating": pl.Float32,
                "timestamp": pl.Int32,
            }
        else:
            raise ValueError(f"Cannot load data with prefix '{file_prefix}'")
        data_path = self._movielens_dir.joinpath(f"{file_prefix}.dat")
        processed_rows = []
        with open(data_path, "r", encoding="ISO-8859-1") as file:
            for line in file:
                row = dict(zip(schema.keys(), line.strip().split("::")))
                processed_rows.append(row)
        return pl.DataFrame(processed_rows, schema=schema)


class Preprocess20MJob(PreprocessJob, ExecutableJobs):
    """
    Download MovieLens 20M data from source, preprocess raw data and save preprocessed data
    """

    def __init__(self):
        job_name = "preprocess-20m"
        PreprocessJob.__init__(self, job_name)
        ExecutableJobs.__init__(self, job_name)

    def _load_data(self, file_prefix: str) -> pl.DataFrame:
        data_path = self._movielens_dir.joinpath(f"{file_prefix}.csv")
        data = pl.read_csv(data_path)
        if file_prefix == "movies":
            return data.rename({"movieId": "movie_id"})
        elif file_prefix == "ratings":
            return data.rename({"userId": "user_id", "movieId": "movie_id"})
        else:
            raise ValueError(f"Cannot load data with prefix '{file_prefix}'")
