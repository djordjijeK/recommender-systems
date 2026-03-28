import pandas as pd

from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent.parent

USERS_DATA_PATH  = _BASE_DIR / "data" / "users.dat"
MOVIES_DATA_PATH = _BASE_DIR / "data" / "movies.dat"
RATINGS_DATA_PATH = _BASE_DIR / "data" / "ratings.dat"


GENDER_MAP: dict[str, int] = {"M": 0, "F": 1}

ALL_GENRES: list[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

_AGE_BOUNDARIES: list[int] = [1, 18, 25, 35, 45, 50, 56]


def get_users(path: Path = USERS_DATA_PATH) -> pd.DataFrame:
    users = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
    )

    users["user_id"] -= 1
    users["gender"] = users["gender"].map(GENDER_MAP)
    users["age"] = users["age"].map(_AGE_BOUNDARIES.index)

    users.drop(columns=["zip_code"], inplace=True)

    return users


def get_movies(path: Path = MOVIES_DATA_PATH) -> pd.DataFrame:
    movies = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    movies["movie_id"] -= 1
    movies["release_year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(int)

    for genre in ALL_GENRES:
        column = "genre_" + genre.lower().replace("-", "_").replace("'", "")
        movies[column] = movies["genres"].str.contains(genre, regex=False).astype(int)

    movies.drop(columns=["title", "genres"], inplace=True)

    return movies


def get_ratings(path: Path = RATINGS_DATA_PATH) -> pd.DataFrame:
    ratings = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    ratings["user_id"] -= 1
    ratings["movie_id"] -= 1

    return ratings


def get_data() -> pd.DataFrame:
    users = get_users()
    movies = get_movies()
    ratings = get_ratings()

    data = ratings.merge(users, on="user_id", how="left")
    data = data.merge(movies, on="movie_id", how="left")

    return data
