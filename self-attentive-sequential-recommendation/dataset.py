import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import random

import torch

from load.movies import get_data
from torch.utils.data import Dataset


def build_dataset(min_interactions: int = 17, val_size: int = 1, test_size: int = 1) -> tuple[dict, dict, dict, int, int]:
    data = get_data()

    while True:
        n_before = len(data)
        
        movie_interactions = data["movie_id"].value_counts()
        data = data[data["movie_id"].isin(movie_interactions[movie_interactions >= min_interactions].index)]
        
        user_interactions = data["user_id"].value_counts()
        data = data[data["user_id"].isin(user_interactions[user_interactions >= min_interactions].index)]

        if len(data) == n_before:
            break

    user_map = {user_id: index + 1 for index, user_id in enumerate(sorted(data["user_id"].unique()))}
    movie_map = {movie_id: index + 1 for index, movie_id in enumerate(sorted(data["movie_id"].unique()))}
    data["user_id"] = data["user_id"].map(user_map)
    data["movie_id"] = data["movie_id"].map(movie_map)

    num_users = len(user_map)
    num_items = len(movie_map)

    data = data.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    data["_rank"] = data.groupby("user_id")["timestamp"].rank(method="first", ascending=False).astype(int)
    
    def _held_out(rank_low: int, rank_high: int) -> dict[int, list[int]]:
        subset = data[(data["_rank"] >= rank_low) & (data["_rank"] <= rank_high)]
        return subset.sort_values("timestamp").groupby("user_id")["movie_id"].apply(list).to_dict()
        
    test_sequences = _held_out(1, test_size)                           
    valid_sequences  = _held_out(test_size + 1, test_size + val_size)

    cutoff = test_size + val_size
    train_sequences: dict[int, list[int]] = data[(data["_rank"] > cutoff)].sort_values("timestamp").groupby("user_id")["movie_id"].apply(list).to_dict()

    return train_sequences, valid_sequences, test_sequences, num_users, num_items


class SelfAttentiveSequentialRecommenderTrainDataset(Dataset):

    def __init__(
        self, 
        train_sequences: dict[int, list[int]], 
        num_items: int, 
        context_length: int = 64
    ) -> None:
        self._train_sequences = train_sequences
        self._context_length = context_length
        self._users = list(train_sequences.keys())
        self._num_items = num_items


    def __len__(self) -> int:
        return len(self._users)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user = self._users[index]
        user_sequence = self._train_sequences[user]

        # input and positive sequences
        input_ids = user_sequence[:-1] 
        positive_target_ids = user_sequence[1:]   

        input_ids = input_ids[-self._context_length:]
        positive_target_ids = positive_target_ids[-self._context_length:]

        # negative sampling
        seen = set(user_sequence)

        negative_target_ids: list[int] = []
        for _ in positive_target_ids:
            negative_target_id = random.randint(1, self._num_items)
            while negative_target_id in seen:
                negative_target_id = random.randint(1, self._num_items)
                
            negative_target_ids.append(negative_target_id)
            seen.add(negative_target_id)

        # sequence padding
        padding_length = self._context_length - len(input_ids)

        # input, positive and negative sequences
        input_sequence = [0] * padding_length + input_ids
        positive_targets_sequence = [0] * padding_length + positive_target_ids
        negative_targets_sequence = [0] * padding_length + negative_target_ids

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "input_sequence": torch.tensor(input_sequence, dtype=torch.long),
            "positive_targets_sequence": torch.tensor(positive_targets_sequence, dtype=torch.long),
            "negative_targets_sequence": torch.tensor(negative_targets_sequence, dtype=torch.long)
        }


class SelfAttentiveSequentialRecommenderEvalDataset(Dataset):

    def __init__(
        self, 
        train_sequences: dict[int, list[int]], 
        target_sequences: dict[int, list[int]], 
        num_items: int, 
        context_length: int = 64, 
        num_negatives: int = 32
    ) -> None:
        self._train_sequences = train_sequences
        self._target_sequences = target_sequences
        
        self._num_items = num_items
        self._context_length = context_length
        self._num_negatives = num_negatives
        
        # only keep users that have both a train history AND targets
        self._users = sorted(set(train_sequences.keys()) & set(target_sequences.keys()))


    def __len__(self) -> int:
        return len(self._users)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user = self._users[index]
        input_sequence = self._train_sequences[user]
        positive_targets_sequence = self._target_sequences[user]

        positive_targets_sequence = positive_targets_sequence[-self._context_length:]

        seen_items = set(input_sequence) | set(positive_targets_sequence)

        negative_targets_sequence: list[int] = []
        while len(negative_targets_sequence) < self._num_negatives:
            negative_id = random.randint(1, self._num_items)
            if negative_id not in seen_items:
                negative_targets_sequence.append(negative_id)
                seen_items.add(negative_id)

        input_sequence = input_sequence[-self._context_length:]
        padding_length = self._context_length - len(input_sequence)
        input_sequence = [0] * padding_length + input_sequence

        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "input_sequence": torch.tensor(input_sequence, dtype=torch.long),
            "positive_targets_sequence": torch.tensor(positive_targets_sequence, dtype=torch.long),
            "negative_targets_sequence": torch.tensor(negative_targets_sequence, dtype=torch.long)
        }
