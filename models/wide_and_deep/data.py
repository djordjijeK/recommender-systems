import torch
import random
import numpy as np

from torch.utils.data import Dataset
from load.movies import get_ratings, get_movies


def build_sequences(min_interactions: int = 5) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], int, int, dict[int, int]]:
    data = get_ratings()

    while True:
        n_before = len(data)

        movie_counts = data["movie_id"].value_counts()
        data = data[data["movie_id"].isin(
            movie_counts[movie_counts >= min_interactions].index
        )]

        user_counts = data["user_id"].value_counts()
        data = data[data["user_id"].isin(
            user_counts[user_counts >= min_interactions].index
        )]

        if len(data) == n_before:
            break

    # IDs start at 1 so that 0 is free for PAD.
    user_map = {user_id: index + 1 for index, user_id in enumerate(sorted(data["user_id"].unique()))}
    movie_map = {movie_id: index + 1 for index, movie_id in enumerate(sorted(data["movie_id"].unique()))}

    data["user_id"] = data["user_id"].map(user_map)
    data["movie_id"] = data["movie_id"].map(movie_map)

    num_users = len(user_map)
    num_movies = len(movie_map)

    data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Rank 1 = most recent, rank 2 = second-most-recent, etc.
    data["_rank"] = (
        data.groupby("user_id")["timestamp"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    test_sequences: dict[int, list[int]] = (
        data[data["_rank"] == 1]
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    valid_sequences: dict[int, list[int]] = (
        data[data["_rank"] == 2]
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    train_sequences: dict[int, list[int]] = (
        data[data["_rank"] >= 3]
        .sort_values("timestamp")
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    return train_sequences, valid_sequences, test_sequences, num_users, num_movies, movie_map


def build_genre_matrix(movie_map: dict[int, int], num_movies: int) -> np.ndarray:
    # Lookup table: genre_matrix[remapped_movie_id] → 18-dim binary genre vector.
    movies_df = get_movies()
    genre_columns = [column for column in movies_df.columns if column.startswith("genre_")]

    genre_matrix = np.zeros((num_movies + 1, len(genre_columns)), dtype=np.float32)

    for original_id, remapped_id in movie_map.items():
        row = movies_df.loc[movies_df["movie_id"] == original_id, genre_columns]
        if not row.empty:
            genre_matrix[remapped_id] = row.values[0]

    return genre_matrix


def build_user_genre_profiles(train_sequences: dict[int, list[int]], genre_matrix: np.ndarray, num_users: int) -> np.ndarray:
    # For each user, average the genre vectors of all movies they watched during training.
    # The result is a normalized genre distribution: what fraction of the user's history
    # belongs to each genre. Averaging keeps the values scale-invariant across users
    # with different activity levels.
    num_genres = genre_matrix.shape[1]
    profiles = np.zeros((num_users + 1, num_genres), dtype=np.float32)

    for user, items in train_sequences.items():
        if items:
            profiles[user] = genre_matrix[items].mean(axis=0)

    return profiles


def _wide_features(user_profile: np.ndarray, movie_genres: np.ndarray) -> np.ndarray:
    # Builds the 54-dim wide feature vector for one (user, movie) pair:
    #   user_profile (18): what genres does the user generally like?
    #   movie_genres (18): what genres does this movie have?
    #   user_profile * movie_genres (18): cross-product - how much does the user like the specific genres THIS movie belongs to?
    return np.concatenate([user_profile, movie_genres, user_profile * movie_genres])


class WideDeepTrainDataset(Dataset):

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        num_movies: int,
        genre_matrix: np.ndarray,
        user_profiles: np.ndarray,
    ) -> None:
        self._num_movies = num_movies
        self._genre_matrix = genre_matrix
        self._user_profiles = user_profiles

        self._movies: list[tuple[int, int]] = []
        self._user_seen: dict[int, set[int]] = {}

        for user, movies in train_sequences.items():
            for movie in movies:
                self._movies.append((user, movie))

            self._user_seen[user] = set(movies)


    def __len__(self) -> int:
        return len(self._movies)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_id, positive_movie = self._movies[index]
        user_seen_movies = self._user_seen[user_id]

        negative_movie = random.randint(1, self._num_movies)
        while negative_movie in user_seen_movies:
            negative_movie = random.randint(1, self._num_movies)

        user_profile = self._user_profiles[user_id]
        positive_movie_wide_features = _wide_features(user_profile, self._genre_matrix[positive_movie])
        negative_movie_wide_features = _wide_features(user_profile, self._genre_matrix[negative_movie])

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "positive_movie": torch.tensor(positive_movie, dtype=torch.long),
            "negative_movie": torch.tensor(negative_movie, dtype=torch.long),
            "positive_movie_wide_features": torch.tensor(positive_movie_wide_features, dtype=torch.float),
            "negative_movie_wide_features": torch.tensor(negative_movie_wide_features, dtype=torch.float),
        }


class WideDeepEvalDataset(Dataset):

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        target_sequences: dict[int, list[int]],
        num_movies: int,
        genre_matrix: np.ndarray,
        user_profiles: np.ndarray,
        num_negatives: int = 100,
    ) -> None:
        self._train_sequences = train_sequences
        self._target_sequences = target_sequences
        self._num_movies = num_movies
        self._genre_matrix = genre_matrix
        self._user_profiles = user_profiles
        self._num_negatives = num_negatives

        self._users = sorted(set(train_sequences.keys()) & set(target_sequences.keys()))
        self._negatives = self._precompute_negatives()


    def __len__(self) -> int:
        return len(self._users)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_id = self._users[index]
        positive_movie = self._target_sequences[user_id][0]
        negative_movies = self._negatives[user_id]

        candidate_ids = [positive_movie] + negative_movies

        user_profile = self._user_profiles[user_id]
        candidate_wide_features = np.stack([
            _wide_features(user_profile, self._genre_matrix[item_id])
            for item_id in candidate_ids
        ])

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "candidate_wide_features": torch.tensor(candidate_wide_features, dtype=torch.float),
        }


    def _precompute_negatives(self) -> dict[int, list[int]]:
        rng = random.Random(1347)

        result = {}
        for user in self._users:
            visited = set(self._train_sequences[user]) | set(self._target_sequences[user])
            negatives = []

            while len(negatives) < self._num_negatives:
                candidate = rng.randint(1, self._num_movies)
                if candidate not in visited:
                    negatives.append(candidate)
                    visited.add(candidate)
            
            result[user] = negatives

        return result
