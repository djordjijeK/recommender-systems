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


class SASRecTrainDataset(Dataset):

    def __init__(
        self,
        train_tokens: dict[int, list[int]],
        num_movies: int,
        context_length: int = 128,
    ) -> None:
        self._tokens = train_tokens
        self._num_movies = num_movies
        self._context_length = context_length
        self._users = list(train_tokens.keys())


    def __len__(self) -> int:
        return len(self._users)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user = self._users[index]
        sequence = self._tokens[user]

        tokens = sequence[:-1]
        positive_ids = sequence[1:]

        tokens = tokens[-self._context_length:]
        positive_ids = positive_ids[-self._context_length:]

        seen = set(sequence)
        negative_ids: list[int] = []
        for _ in positive_ids:
            negative_id = random.randint(1, self._num_movies)
            while negative_id in seen:
                negative_id = random.randint(1, self._num_movies)
            negative_ids.append(negative_id)
            seen.add(negative_id)

        padding_length = self._context_length - len(tokens)
        tokens = [0] * padding_length + tokens
        positive_ids = [0] * padding_length + positive_ids
        negative_ids = [0] * padding_length + negative_ids

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "positive_ids": torch.tensor(positive_ids, dtype=torch.long),
            "negative_ids": torch.tensor(negative_ids, dtype=torch.long),
        }


class SASRecEvalDataset(Dataset):

    def __init__(
        self,
        train_tokens: dict[int, list[int]],
        target_tokens: dict[int, list[int]],
        num_movies: int,
        context_length: int = 128,
        num_negatives: int = 100,
    ) -> None:
        self._train_tokens = train_tokens
        self._target_tokens = target_tokens
        self._num_movies = num_movies
        self._context_length = context_length
        self._num_negatives = num_negatives

        self._users = sorted(set(train_tokens.keys()) & set(target_tokens.keys()))
        self._negatives = self._precompute_negatives()


    def __len__(self) -> int:
        return len(self._users)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user = self._users[index]
        tokens = self._train_tokens[user]

        tokens = tokens[-self._context_length:]
        padding_length = self._context_length - len(tokens)
        tokens = [0] * padding_length + tokens

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "positive_movie": torch.tensor(self._target_tokens[user][0], dtype=torch.long),
            "negative_movies": torch.tensor(self._negatives[user], dtype=torch.long),
        }


    def _precompute_negatives(self) -> dict[int, list[int]]:
        rng = random.Random(1347)

        result = {}
        for user in self._users:
            seen = set(self._train_tokens[user]) | set(self._target_tokens[user])
            
            negatives = []
            while len(negatives) < self._num_negatives:
                s = rng.randint(1, self._num_movies)
                if s not in seen:
                    negatives.append(s)
                    seen.add(s)
            
            result[user] = negatives

        return result
