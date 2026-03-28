import math
import random
import torch
import logging
import pandas as pd
import torch.nn.functional as F

from dataclasses import dataclass
from torch import nn
from torch.utils.data import Dataset, DataLoader

import config
from load.movies import get_ratings


logger = logging.getLogger("model:matrix-factorization")


@dataclass
class Config:
    embedding_dim: int = 64
    num_train_negatives: int = 4
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 1e-3
    num_negatives: int = 100
    device: str = "mps" if torch.mps.is_available() else "cpu"


def leave_one_out_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values("timestamp")
    test = ratings.groupby("user_id").tail(1)
    train = ratings.drop(test.index)
    return train, test


class TrainDataset(Dataset):

    def __init__(self, train: pd.DataFrame, all_movie_ids: list[int], num_negatives: int = 4) -> None:
        self._all_movie_ids = all_movie_ids
        self._num_negatives = num_negatives

        self._pairs: list[tuple[int, int]] = list(zip(train["user_id"], train["movie_id"]))
        self._user_seen: dict[int, set[int]] = (
            train.groupby("user_id")["movie_id"].apply(set).to_dict()
        )


    def __len__(self) -> int:
        return len(self._pairs)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        user_id, positive_movie = self._pairs[index]
        seen = self._user_seen[user_id]

        negative_movies: list[int] = []
        while len(negative_movies) < self._num_negatives:
            candidate = random.choice(self._all_movie_ids)
            if candidate not in seen:
                negative_movies.append(candidate)

        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "positive_movie": torch.tensor(positive_movie, dtype=torch.long),
            "negative_movies": torch.tensor(negative_movies, dtype=torch.long),
        }


class MatrixFactorization(nn.Module):

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int) -> None:
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(user_ids)
        movie_emb = self.movie_embeddings(movie_ids)

        dot = (user_emb * movie_emb).sum(dim=1)
        bias = self.user_bias(user_ids).squeeze(-1) + self.movie_bias(movie_ids).squeeze(-1) + self.global_bias

        return dot + bias


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test: pd.DataFrame,
    train: pd.DataFrame,
    all_movie_ids: list[int],
    device: str,
    num_negatives: int = 100,
) -> dict[str, float]:
    model.eval()
    rng = random.Random(1347)
    train_user_movies = train.groupby("user_id")["movie_id"].apply(set).to_dict()

    user_ids_list = []
    candidates_list = []

    for row in test.itertuples():
        user_id, positive_movie_id = row.user_id, row.movie_id
        seen = train_user_movies.get(user_id, set()) | {positive_movie_id}

        negative_movies: list[int] = []
        while len(negative_movies) < num_negatives:
            candidate = rng.choice(all_movie_ids)
            if candidate not in seen:
                negative_movies.append(candidate)
                seen.add(candidate)

        user_ids_list.append(user_id)
        candidates_list.append([positive_movie_id] + negative_movies)

    # Single batched forward pass over all users: [num_users, 1 + num_negatives]
    candidates = torch.tensor(candidates_list, dtype=torch.long, device=device)
    user_ids = torch.tensor(user_ids_list, dtype=torch.long, device=device).unsqueeze(1).expand_as(candidates)

    scores = model(user_ids.reshape(-1), candidates.reshape(-1)).reshape(candidates.shape)

    ranks = (scores[:, 1:] >= scores[:, 0:1]).sum(dim=1).float() + 1

    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        hit = (ranks <= k).float()
        metrics[f"HR@{k}"] = hit.mean().item()
        if k != 1:
            metrics[f"NDCG@{k}"] = (hit / torch.log2(ranks + 1)).mean().item()
    metrics["MRR"] = (1.0 / ranks).mean().item()

    return metrics


if __name__ == "__main__":
    cfg = Config()

    ratings = get_ratings()

    num_users = ratings["user_id"].max() + 1
    num_movies = ratings["movie_id"].max() + 1

    train, test = leave_one_out_split(ratings)
    all_movie_ids = ratings["movie_id"].unique().tolist()

    train_dataset = TrainDataset(train, all_movie_ids, num_negatives=cfg.num_train_negatives)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size)

    model = MatrixFactorization(num_users, num_movies, cfg.embedding_dim).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)

    logger.info(f"Training Matrix Factorization on {cfg.device}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("=" * 100)

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            user_ids = batch["user_id"].to(cfg.device)                  # [B]
            positive_movies = batch["positive_movie"].to(cfg.device)    # [B]
            negative_movies = batch["negative_movies"].to(cfg.device)   # [B, num_train_negatives]

            batch_size = user_ids.size(0)
            num_neg = negative_movies.size(1)

            positive_scores = model(user_ids, positive_movies)  # [B]

            user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_neg).reshape(-1)
            negative_scores = model(user_ids_expanded, negative_movies.reshape(-1))  # [B * num_neg]

            logits = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([torch.ones(batch_size, device=cfg.device), torch.zeros(batch_size * num_neg, device=cfg.device)])

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        metrics = evaluate(model, test, train, all_movie_ids, cfg.device, cfg.num_negatives)
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())

        logger.info(
            f"Epoch {epoch:>3d}/{cfg.num_epochs:<3d} | "
            f"Train Loss: {epoch_loss / steps_per_epoch:>7.4f} | "
            f"{metrics_str}"
        )
