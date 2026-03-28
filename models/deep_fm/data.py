import torch
import random
import numpy as np

from torch.utils.data import Dataset
from load.movies import get_ratings, get_users, get_movies


def build_sequences(min_interactions: int = 5) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], int, int, dict[int, int], dict[int, int]]:
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

    return train_sequences, valid_sequences, test_sequences, num_users, num_movies, user_map, movie_map


def build_user_features(user_map: dict[int, int], num_users: int) -> np.ndarray:
    users_df = get_users()
    user_features = np.zeros((num_users + 1, 3), dtype=np.int64)

    for original_id, remapped_id in user_map.items():
        row = users_df.loc[users_df["user_id"] == original_id]
        if not row.empty:
            user_features[remapped_id] = [
                row["gender"].values[0],
                row["age"].values[0],
                row["occupation"].values[0],
            ]

    return user_features


def build_movie_features(movie_map: dict[int, int], num_movies: int) -> np.ndarray:
    movies_df = get_movies()
    genre_columns = [column for column in movies_df.columns if column.startswith("genre_")]

    movie_features = np.zeros((num_movies + 1, len(genre_columns)), dtype=np.int64)

    for original_id, remapped_id in movie_map.items():
        row = movies_df.loc[movies_df["movie_id"] == original_id, genre_columns]
        if not row.empty:
            movie_features[remapped_id] = row.values[0]

    return movie_features


def _build_feature_vector(
    user_id: int,
    movie_id: int,
    user_features: np.ndarray,
    movie_features: np.ndarray,
) -> list[int]:
    # [user_id, movie_id, gender, age, occupation, genre_1, ..., genre_18]
    return [user_id, movie_id] + user_features[user_id].tolist() + movie_features[movie_id].tolist()


class DeepFMTrainDataset(Dataset):

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        num_movies: int,
        user_features: np.ndarray,
        movie_features: np.ndarray,
        num_negatives: int = 4,
    ) -> None:
        self._num_movies = num_movies
        self._num_negatives = num_negatives
        self._user_features = user_features
        self._movie_features = movie_features

        self._pairs: list[tuple[int, int]] = []
        self._user_seen: dict[int, set[int]] = {}

        for user, movies in train_sequences.items():
            for movie in movies:
                self._pairs.append((user, movie))

            self._user_seen[user] = set(movies)


    def __len__(self) -> int:
        return len(self._pairs)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_id, positive_movie = self._pairs[index]
        user_seen_movies = self._user_seen[user_id]

        negative_movies: list[int] = []
        while len(negative_movies) < self._num_negatives:
            candidate = random.randint(1, self._num_movies)
            if candidate not in user_seen_movies:
                negative_movies.append(candidate)

        positive_features = _build_feature_vector(user_id, positive_movie, self._user_features, self._movie_features)

        negative_features = [
            _build_feature_vector(user_id, negative_movie, self._user_features, self._movie_features)
            for negative_movie in negative_movies
        ]

        return {
            "positive_features": torch.tensor(positive_features, dtype=torch.long),
            "negative_features": torch.tensor(negative_features, dtype=torch.long),
        }


class DeepFMEvalDataset(Dataset):

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        target_sequences: dict[int, list[int]],
        num_movies: int,
        user_features: np.ndarray,
        movie_features: np.ndarray,
        num_negatives: int = 100,
    ) -> None:
        self._train_sequences = train_sequences
        self._target_sequences = target_sequences
        self._num_movies = num_movies
        self._user_features = user_features
        self._movie_features = movie_features
        self._num_negatives = num_negatives

        self._users = sorted(set(train_sequences.keys()) & set(target_sequences.keys()))
        self._negatives = self._precompute_negatives()


    def __len__(self) -> int:
        return len(self._users)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_id = self._users[index]
        positive_movie = self._target_sequences[user_id][0]
        negative_movies = self._negatives[user_id]

        candidates = [positive_movie] + negative_movies

        candidate_features = [
            _build_feature_vector(user_id, movie_id, self._user_features, self._movie_features)
            for movie_id in candidates
        ]

        return {
            "candidate_features": torch.tensor(candidate_features, dtype=torch.long),
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
