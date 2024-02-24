import os
import pathlib
import logging
from dataclasses import dataclass

logging.basicConfig(
    format="%(asctime)s (%(module)s.%(funcName)s) : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)


@dataclass
class DataConfig:
    source_url: str
    movies_schema: dict[str, str]
    users_schema: dict[str, str]
    ratings_schema: dict[str, str]
    min_interactions: int

    @property
    def data_dir(self):
        return pathlib.Path(f"{os.getcwd()}/data")

    @property
    def raw_data_dir(self):
        return self.data_dir.joinpath("ml-1m")

    @property
    def movielens_dir(self):
        return self.data_dir.joinpath("movielens")


data_config = DataConfig(
    source_url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    min_interactions=20,
    movies_schema={
        "movie_id": "int",
        "title": "varchar",
        "genres": "varchar",
    },
    users_schema={
        "user_id": "int",
        "gender": "varchar",
        "age": "varchar",
        "occupation": "varchar",
        "zip_code": "varchar",
    },
    ratings_schema={
        "user_id": "int",
        "movie_id": "int",
        "rating": "short",
        "timestamp": "int",
    }
)
