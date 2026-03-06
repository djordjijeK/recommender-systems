import logging
import pandas as pd

from pathlib import Path
from config import MOVIES_DATA_PATH, RATINGS_DATA_PATH


logger = logging.getLogger("utils")


def get_data() -> pd.DataFrame:
    base_path = Path(__file__).resolve().parents[1]
    movies_data_path  = Path(f"{base_path}/{MOVIES_DATA_PATH}")
    ratings_data_path = Path(f"{base_path}/{RATINGS_DATA_PATH}")

    logger.debug(f"Loading data...")

    movies_data_frame = pd.read_csv(movies_data_path)
    ratings_data_frame = pd.read_csv(ratings_data_path)

    data_frame = pd.merge(ratings_data_frame, movies_data_frame, on="movieId")
    data_frame = data_frame[["userId", "movieId", "title", "rating"]]

    data_frame = data_frame.rename(columns={
        "userId": "user_id",
        "movieId": "movie_id"
    })

    logger.debug(f"Data loading finished")

    return data_frame