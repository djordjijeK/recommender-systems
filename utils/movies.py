from pathlib import Path

import pandas as pd


_BASE_DIR = Path(__file__).resolve().parent.parent

USERS_DATA_PATH  = _BASE_DIR / "data" / "users.dat"
MOVIES_DATA_PATH = _BASE_DIR / "data" / "movies.dat"
RATINGS_DATA_PATH = _BASE_DIR / "data" / "ratings.dat"


AGE_MAP: dict[int, str] = {
    0: "1-17",
    1: "18-24",
    2: "25-34",
    3: "35-44",
    4: "45-49",
    5: "50-55",
    6: "56-100",
}

OCCUPATION_MAP: dict[int, str] = {
    0:  "other",
    1:  "academic",
    2:  "artist",
    3:  "clerical",
    4:  "grad_student",
    5:  "customer_service",
    6:  "healthcare",
    7:  "executive",
    8:  "farmer",
    9:  "homemaker",
    10: "k12_student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales",
    15: "scientist",
    16: "self_employed",
    17: "engineer",
    18: "tradesman",
    19: "unemployed",
    20: "writer",
}

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
        col = "genre_" + genre.lower().replace("-", "_").replace("'", "")
        movies[col] = movies["genres"].str.contains(genre, regex=False).astype(int)

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

    ratings["user_id"]  -= 1
    ratings["movie_id"] -= 1

    return ratings

def get_data() -> pd.DataFrame:
    users   = get_users()
    movies  = get_movies()
    ratings = get_ratings()

    data = ratings.merge(users,  on="user_id",  how="left")
    data = data.merge(movies, on="movie_id", how="left")

    return data

def get_train_val_test_split(data: pd.DataFrame, validation_size: int = 5, test_size: int = 10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train / validation / test sets.

    Splitting strategy: for each user, the *N* most-recent interactions
    (by timestamp) go to test, the next *M* to validation, and everything
    older than that to train.

    Args:
        data: Merged dataset returned by :func:`get_data`.
        val_size: Number of interactions per user held out for validation.
        test_size: Number of most-recent interactions per user held out for test.

    Returns:
        A ``(train, val, test)`` tuple of non-overlapping DataFrames.
    """
    data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    data["_rank"] = (
        data.groupby("user_id")["timestamp"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    test  = data[data["_rank"] <= test_size]
    val   = data[(data["_rank"] > test_size) & (data["_rank"] <= test_size + validation_size)]
    train = data[data["_rank"] > test_size + validation_size]

    drop = ["_rank"]
    return (
        train.drop(columns=drop),
        val.drop(columns=drop),
        test.drop(columns=drop),
    )
