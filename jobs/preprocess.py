import json
import shutil
import requests
import logging
import duckdb
from jobs.config import ExecutableJobs, data_config


class PreProcessJob(ExecutableJobs):
    """
    preprocess job
    """

    def __init__(self):
        super(PreProcessJob, self).__init__(name="preprocess")
        self._db = None

    def execute(self):
        self._db = duckdb.connect()
        logging.info("download data from source_url")
        self.download_data(data_config.source_url)

        logging.info("preprocess raw data into 'newline_delimited' json format")
        self.process_data("users", data_config.users_schema)
        self.process_data("movies", data_config.movies_schema)
        self.process_data("ratings", data_config.ratings_schema)

        logging.info("register preprocessed data into duckdb")
        data_dir = data_config.data_dir
        self.register_data(f"{data_dir}/movies.json", "movies", data_config.movies_schema)
        self.register_data(f"{data_dir}/users.json", "users", data_config.users_schema)
        self.register_data(f"{data_dir}/ratings.json", "ratings", data_config.ratings_schema)

        logging.info("filter out ratings of/on user/movie to alleviate sparsity of ratings data")
        self.filter_data(data_config.min_interactions)

        logging.info("convert user_id, movie_id in ratings into corresponding index")
        self.convert_filtered_data()

        logging.info("export tables registered in database")
        self._db.execute(f"EXPORT DATABASE {data_config.movielens_dir}")

    @staticmethod
    def download_data(source_url: str):
        """
        :param source_url:  URL to MovieLens 1 Million data(ml-1m.zip)
        """
        data_config.data_dir.mkdir(exist_ok=True, parents=True)
        with open(data_config.data_dir.joinpath("ml-1m.zip"), "wb") as file:
            response = requests.get(source_url)
            file.write(response.content)
        shutil.unpack_archive(
            filename=data_config.data_dir.joinpath("ml-1m.zip"),
            extract_dir=data_config.data_dir,
            format="zip",
        )

    @staticmethod
    def process_data(file_prefix: str, column_names: list[str]):
        """
        Within context manager, special encoding is selected to avoid UnicodeDecodeError
        :param file_prefix: file name without extension
        :param column_names: list of column names in order which data is aligned
        """
        processed_rows = []
        raw_data_path = data_config.raw_data_dir.joinpath(f"{file_prefix}.dat")
        with open(raw_data_path, "r", encoding="ISO-8859-1") as raw_data:
            for line in raw_data:
                row = dict(zip(column_names, line.strip().split("::")))
                processed_rows.append(json.dumps(row))
        with open(data_config.data_dir.joinpath(f"{file_prefix}.json"), "w") as processed_data:
            processed_data.write("\n".join(processed_rows))

    def register_data(self, file_path: str, name: str, data_schema: dict[str, str]):
        """
        :param file_path: path to processed raw data in 'newline_delimited' format
        :param name: name of table
        :param data_schema: schema of corresponding table
        """
        schema_value = ", ".join([" ".join(item) for item in data_schema.items()])
        queries = [
            f"DROP TABLE IF EXISTS {name}",
            f"CREATE TABLE {name} ({schema_value})",
            f"COPY {name} FROM '{file_path}'"
        ]
        for query in queries:
            self._db.execute(query)

    def filter_data(self, min_interactions: int):
        """
        Filter out users/movies with less interaction to reduce sparsity as in Ch6 of https://arxiv.org/pdf/2006.15516.pdf
        (reference: https://huggingface.co/datasets/reczoo/Movielens1M_m1)
        :param min_interactions: exclude user/item with number of interactions is less than this value
        """
        queries = [
            """
                DROP TABLE IF EXISTS filtered_ratings
            """,
            f"""
                CREATE TABLE filtered_ratings AS
                SELECT *
                FROM ratings
                QUALIFY count(*) OVER (PARTITION BY user_id) >= {min_interactions}
                    AND count(*) OVER (PARTITION BY movie_id) >= {min_interactions}
            """
        ]
        for query in queries:
            self._db.execute(query)

    def convert_filtered_data(self):
        """
        convert distinct user_id and movie_id in filtered ratings data to use them as corresponding embedding index
        """
        queries = [
            """
                DROP TABLE IF EXISTS user_index_map
            """,
            """
                CREATE TABLE user_index_map AS
                SELECT user_id, 
                       row_number() OVER (ORDER BY user_id) - 1 AS user_index
                FROM filtered_ratings
                GROUP BY user_id
            """,
            """
                DROP TABLE IF EXISTS movie_index_map
            """,
            """
                CREATE TABLE movie_index_map AS
                SELECT movie_id, 
                       row_number() OVER (ORDER BY movie_id) - 1 AS movie_index
                FROM filtered_ratings
                GROUP BY movie_id
            """,
            """
                DROP TABLE IF EXISTS converted_ratings
            """,
            """
                CREATE TABLE converted_ratings AS
                SELECT ui.user_index,
                       mi.movie_index,
                       fr.timestamp
                FROM filtered_ratings fr
                    JOIN user_index_map ui on fr.user_id = ui.user_id
                    JOIN movie_index_map mi on fr.movie_id = mi.movie_id
            """
        ]
        for query in queries:
            self._db.execute(query)
