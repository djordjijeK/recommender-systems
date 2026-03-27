import torch
import random

from load.movies import get_ratings
from torch.utils.data import Dataset


def build_sequences(min_interactions: int = 5) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], int, int]:
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

    return train_sequences, valid_sequences, test_sequences, num_users, num_movies


class NCFTrainDataset(Dataset):

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        num_movies: int,
        num_negatives: int = 4,
    ) -> None:
        self._num_movies = num_movies
        self._num_negatives = num_negatives

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

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "positive_movie": torch.tensor(positive_movie, dtype=torch.long),
            "negative_movies": torch.tensor(negative_movies, dtype=torch.long),
        }


class NCFEvalDataset(Dataset):

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        target_sequences: dict[int, list[int]],
        num_movies: int,
        num_negatives: int = 100,
    ) -> None:
        self._train_sequences = train_sequences
        self._target_sequences = target_sequences
        self._num_movies = num_movies
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

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
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
