import math
import torch
import config
import logging
import torch.nn.functional as F

from dataclasses import dataclass
from .model import WideDeep
from .data import (
    build_sequences,
    build_genre_matrix,
    build_user_genre_profiles,
    WideDeepTrainDataset,
    WideDeepEvalDataset,
)
from torch.utils.data import DataLoader


logger = logging.getLogger("model:Wide&Deep")


@dataclass
class Config:
    min_interactions: int = 5

    embedding_dim: int = 256
    wide_dim: int = 54  # user_profile (18) + item_genres (18) + cross_product (18)
    hidden_layers: tuple[int, ...] = (512, 256, 128, 16)
    dropout: float = 0.2

    batch_size: int = 512
    num_epochs: int = 100
    learning_rate: float = 1e-3

    num_negatives: int = 100

    device: str = "mps" if torch.mps.is_available() else "cpu"


@torch.no_grad()
def evaluate(
    model: WideDeep,
    data_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    all_ranks: list[torch.Tensor] = []

    for batch_data in data_loader:
        user_ids = batch_data["user_id"].to(device)
        candidate_ids = batch_data["candidate_ids"].to(device)
        candidate_wide = batch_data["candidate_wide_features"].to(device)

        scores = model.score_candidates(user_ids, candidate_ids, candidate_wide)

        positive_score = scores[:, 0:1]
        negative_scores = scores[:, 1:]

        rank = (negative_scores >= positive_score).sum(dim=1) + 1
        all_ranks.append(rank.cpu())

    ranks = torch.cat(all_ranks).float()

    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        hit = (ranks <= k).float()
        metrics[f"HR@{k}"] = hit.mean().item()
        if k != 1:
            metrics[f"NDCG@{k}"] = (hit / torch.log2(ranks + 1)).mean().item()

    metrics["MRR"] = (1.0 / ranks).mean().item()

    return metrics


def train(
    model: WideDeep,
    config: Config,
    train_dataset: WideDeepTrainDataset,
    valid_dataset: WideDeepEvalDataset,
) -> WideDeep:
    device = torch.device(config.device)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate)

    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)

    logger.info(f"Training Wide&Deep on {device}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("=" * 100)

    for epoch in range(1, config.num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        for train_batch in train_data_loader:
            user_ids = train_batch["user_id"].to(device)
            positive_movie = train_batch["positive_movie"].to(device)
            negative_movie = train_batch["negative_movie"].to(device)
            positive_wide_features = train_batch["positive_movie_wide_features"].to(device)
            negative_wide_feature = train_batch["negative_movie_wide_features"].to(device)

            positive_scores = model(user_ids, positive_movie, positive_wide_features)
            negative_scores = model(user_ids, negative_movie, negative_wide_feature)

            logits = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)])

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        validation_metrics = evaluate(model, valid_data_loader, config.device)
        metrics = " | ".join(f"{k}: {v:.4f}" for k, v in validation_metrics.items())

        logger.info(
            f"Epoch {epoch:>3d}/{config.num_epochs:<3d} | "
            f"Train Loss: {epoch_loss / steps_per_epoch:>7.4f} | "
            f"{metrics}"
        )

    return model


if __name__ == "__main__":
    config = Config()

    train_sequences, valid_sequences, test_sequences, num_users, num_movies, movie_map = build_sequences(
        min_interactions=config.min_interactions
    )

    genre_matrix = build_genre_matrix(movie_map, num_movies)
    user_profiles = build_user_genre_profiles(train_sequences, genre_matrix, num_users)

    train_dataset = WideDeepTrainDataset(
        train_sequences=train_sequences,
        num_movies=num_movies,
        genre_matrix=genre_matrix,
        user_profiles=user_profiles,

    )

    valid_dataset = WideDeepEvalDataset(
        train_sequences=train_sequences,
        target_sequences=valid_sequences,
        num_movies=num_movies,
        genre_matrix=genre_matrix,
        user_profiles=user_profiles,
        num_negatives=config.num_negatives
    )

    model = WideDeep(
        num_users=num_users,
        num_movies=num_movies,
        wide_dim=config.wide_dim,
        embedding_dimension=config.embedding_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    ).to(config.device)

    model = train(model, config, train_dataset, valid_dataset)
 
    for user in train_sequences:
        if user in valid_sequences:
            train_sequences[user] += valid_sequences[user]

    user_profiles = build_user_genre_profiles(train_sequences, genre_matrix, num_users)

    test_dataset = WideDeepEvalDataset(
        train_sequences=train_sequences,
        target_sequences=test_sequences,
        num_movies=num_movies,
        genre_matrix=genre_matrix,
        user_profiles=user_profiles,
        num_negatives=config.num_negatives,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
    )

    test_metrics = evaluate(model, test_data_loader, config.device)
    metrics = " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())

    logger.info(f"Test metrics - {metrics}")
