import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))


import torch
import random

from load.movies import get_ratings
from torch.utils.data import Dataset


def build_dataset(min_interactions: int = 5) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]], int, int]:
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
    user_map  = {user_id: index + 1 for index, user_id in enumerate(sorted(data["user_id"].unique()))}
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
    valid_sequences: dict[int, list[int]]= (
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


class BERT4RecTrainDataset(Dataset):
 
    def __init__(
        self,
        train_tokens: dict[int, list[int]],
        num_movies: int,
        context_length: int = 200,
        mask_prob: float = 0.2,
        force_last_item_mask_prob: float = 0.5,
    ) -> None:
        self._tokens = train_tokens
        self._num_movies = num_movies
        self._context_length = context_length
        self._mask_prob = mask_prob

        self._force_last_mask_prob = force_last_item_mask_prob

        self._mask_token = self._num_movies + 1 
        self._users = list(train_tokens.keys())
 

    def __len__(self) -> int:
        return len(self._users)
 

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user = self._users[index]
        tokens = self._tokens[user]
 
        tokens = tokens[-self._context_length:]
 
        if random.random() < self._force_last_mask_prob:
            tokens, labels = self._mask_last_item(tokens)
        else:
            tokens, labels = self._random_mask(tokens)
 
        pading_length = self._context_length - len(tokens)
        tokens  = [0] * pading_length + tokens
        labels  = [0] * pading_length + labels
 
        return {
            "user_id": torch.tensor(user,   dtype=torch.long),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

 
    def _random_mask(self, sequence: list[int]) -> tuple[list[int], list[int]]:
        tokens: list[int] = []
        labels: list[int] = []
 
        for item in sequence:
            if random.random() < self._mask_prob:
                tokens.append(self._mask_token)
                labels.append(item)
            else:
                tokens.append(item)
                labels.append(0)
 
        if all(label == 0 for label in labels):
            idx = random.randrange(len(tokens))
            labels[idx] = tokens[idx]
            tokens[idx] = self._mask_token
 
        return tokens, labels
 

    def _mask_last_item(self, sequence: list[int]) -> tuple[list[int], list[int]]:
        tokens = list(sequence)
        labels = [0] * len(sequence)
 
        last_idx = len(tokens) - 1
        labels[last_idx] = tokens[last_idx]
        tokens[last_idx] = self._mask_token
 
        return tokens, labels


class BERT4RecEvalDataset(Dataset):
 
    def __init__(
        self,
        train_tokens:  dict[int, list[int]],
        target_tokens: dict[int, list[int]],
        num_movies: int,
        context_length: int = 200,
        num_negatives: int  = 100,
    ) -> None:
        self._train_tokens = train_tokens
        self._target_tokens = target_tokens
        self._num_movies = num_movies
        self._context_length = context_length
        self._num_negatives = num_negatives

        self._mask_token = num_movies + 1
 
        # Only evaluate users that have both a training history and a target
        self._users = sorted(set(train_tokens.keys()) & set(target_tokens.keys()))
 

    def __len__(self) -> int:
        return len(self._users)
 

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user = self._users[index]
        tokens = self._train_tokens[user]
        positive_target_tokens = self._target_tokens[user]
  
        tokens = tokens[-( self._context_length - 1):] + [self._mask_token]
        padding_length = self._context_length - len(tokens)
        tokens = [0] * padding_length + tokens
 
        seen = set(self._train_tokens[user]) | set(positive_target_tokens)
        negative_tokens: list[int] = []
        while len(negative_tokens) < self._num_negatives:
            negative_sample = random.randint(1, self._num_movies)
            if negative_sample not in seen:
                negative_tokens.append(negative_sample)
                seen.add(negative_sample)
  
        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "positive_movie": torch.tensor(positive_target_tokens[0], dtype=torch.long),
            "negative_movies": torch.tensor(negative_tokens, dtype=torch.long)
        }
 