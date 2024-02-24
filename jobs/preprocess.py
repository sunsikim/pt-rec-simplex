import json
import shutil
import requests
import logging
import duckdb
from jobs.config import data_config


def execute():
    logger = logging.getLogger(__name__)
    db = duckdb.connect()

    logger.info("download data from source_url")
    download_data(data_config.source_url)

    logger.info("preprocess raw data into 'newline_delimited' json format")
    process_data("users", data_config.users_schema)
    process_data("movies", data_config.movies_schema)
    process_data("ratings", data_config.ratings_schema)

    logger.info("register preprocessed data into duckdb")
    data_dir = data_config.data_dir
    register_data(f"{data_dir}/movies.json", "movies", data_config.movies_schema, db)
    register_data(f"{data_dir}/users.json", "users", data_config.users_schema, db)
    register_data(f"{data_dir}/ratings.json", "ratings", data_config.ratings_schema, db)

    logger.info("filter out ratings of/on user/movie to alleviate sparsity of ratings data")
    filter_data(data_config.min_interactions, db)

    logger.info("convert user_id, movie_id in ratings into corresponding index")
    convert_filtered_data(db)

    logger.info("split converted data into train, validation, test dataset")
    split_data(db)

    logger.info("export tables registered in database")
    db.execute(f"EXPORT DATABASE {data_config.movielens_dir}")


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


def register_data(
    file_path: str,
    name: str,
    data_schema: dict[str, str],
    db: duckdb.DuckDBPyConnection,
):
    """
    :param file_path: path to processed raw data in 'newline_delimited' format
    :param name: name of table
    :param data_schema: schema of corresponding table
    :param db: duckdb connection
    """
    schema_value = ", ".join([" ".join(item) for item in data_schema.items()])
    queries = [
        f"DROP TABLE IF EXISTS {name}",
        f"CREATE TABLE {name} ({schema_value})",
        f"COPY {name} FROM '{file_path}'"
    ]
    for query in queries:
        db.execute(query)


def filter_data(min_interactions: int, db: duckdb.DuckDBPyConnection):
    """
    Filter out users/movies with less interaction to reduce sparsity as in Ch6 of https://arxiv.org/pdf/2006.15516.pdf
    (reference: https://huggingface.co/datasets/reczoo/Movielens1M_m1)
    :param min_interactions: exclude user/item with number of interactions is less than this value
    :param db: duckdb where dataset is registered
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
        db.execute(query)


def convert_filtered_data(db: duckdb.DuckDBPyConnection):
    """
    convert distinct user_id and movie_id in filtered ratings data to use them as corresponding embedding index
    :param db: duckdb where dataset is registered
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
        db.execute(query)


def split_data(db: duckdb.DuckDBPyConnection):
    """
    test data : model will be evaluated by how it predicts the latest movie that each user watched
    feed data : separated into 2 parts
        - validation : randomly sample 10% data from each user
        - training : remaining data
    :param db: duckdb where dataset is registered
    """
    query = """
        SELECT *, 
               row_number() OVER (PARTITION BY user_index ORDER BY timestamp DESC) AS interaction_order,
               count(*) OVER (PARTITION BY user_index) - 1 AS rated_movies_count  -- subtract 1 to exclude test data
        FROM converted_ratings
    """
    ratings_stat = db.execute(query).arrow()

    query = """
        SELECT user_index,
               movie_index,
               row_number() OVER (PARTITION BY user_index ORDER BY random()) random_index,
               0.1 * rated_movies_count AS validation_index_ub
        FROM ratings_stat
        WHERE interaction_order > 1 -- interaction_order = 1 means test data 
    """
    feed_data = db.execute(query).arrow()

    queries = [
        """
            DROP TABLE IF EXISTS test_data
        """,
        """
            CREATE TABLE test_data AS
            SELECT user_index, movie_index
            FROM ratings_stat
            WHERE interaction_order = 1
            ORDER BY user_index
        """,
        """
            DROP TABLE IF EXISTS validation_data
        """,
        """
            CREATE TABLE validation_data AS
            SELECT user_index, movie_index
            FROM feed_data
            WHERE random_index <= validation_index_ub
            ORDER BY user_index
        """,
        """
            DROP TABLE IF EXISTS train_data
        """,
        """
            CREATE TABLE train_data AS
            SELECT user_index, movie_index
            FROM feed_data
            WHERE random_index > validation_index_ub
            ORDER BY user_index
        """
    ]
    for query in queries:
        db.execute(query)
