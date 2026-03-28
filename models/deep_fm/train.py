import math
import torch
import config
import logging
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader

from .model import DeepFM
from .data import (
    build_sequences,
    build_user_features,
    build_movie_features,
    DeepFMTrainDataset,
    DeepFMEvalDataset,
)


logger = logging.getLogger("model:DeepFM")


@dataclass
class Config:
    min_interactions: int = 5

    embedding_dim: int = 32
    hidden_layers: tuple[int, ...] = (512, 256, 128, 64)
    dropout: float = 0.2

    num_train_negatives: int = 8

    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 1e-3

    num_negatives: int = 100

    device: str = "mps" if torch.mps.is_available() else "cpu"


@torch.no_grad()
def evaluate(model: DeepFM, data_loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_ranks: list[torch.Tensor] = []

    for batch_data in data_loader:
        candidate_features = batch_data["candidate_features"].to(device)

        scores = model.score_candidates(candidate_features)

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


def train(model: DeepFM,config: Config, train_dataset: DeepFMTrainDataset, valid_dataset: DeepFMEvalDataset) -> DeepFM:
    device = torch.device(config.device)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate)

    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)

    logger.info(f"Training DeepFM on {device}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("=" * 100)

    for epoch in range(1, config.num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        for train_batch in train_data_loader:
            positive_features = train_batch["positive_features"].to(device)   
            negative_features = train_batch["negative_features"].to(device)    

            batch_size = positive_features.size(0)
            num_negatives = negative_features.size(1)

            positive_scores = model(positive_features)  
            negative_scores = model(negative_features.reshape(batch_size * num_negatives, -1)).reshape(batch_size, num_negatives)

            logits = torch.cat([positive_scores, negative_scores.reshape(-1)])
            labels = torch.cat([torch.ones_like(positive_scores), torch.zeros(batch_size * num_negatives, device=device)])

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

    train_sequences, valid_sequences, test_sequences, num_users, num_movies, user_map, movie_map = build_sequences(
        min_interactions=config.min_interactions
    )

    user_features = build_user_features(user_map, num_users)
    movie_features = build_movie_features(movie_map, num_movies)

    # field_dims: [user_id, movie_id, gender, age, occupation, genre_1, ..., genre_18]
    field_dims = [
        num_users + 1,
        num_movies + 1,
        int(user_features[:, 0].max()) + 1,
        int(user_features[:, 1].max()) + 1,
        int(user_features[:, 2].max()) + 1,
    ] + [2] * movie_features.shape[1]

    train_dataset = DeepFMTrainDataset(
        train_sequences=train_sequences,
        num_movies=num_movies,
        user_features=user_features,
        movie_features=movie_features,
        num_negatives=config.num_train_negatives,
    )

    valid_dataset = DeepFMEvalDataset(
        train_sequences=train_sequences,
        target_sequences=valid_sequences,
        num_movies=num_movies,
        user_features=user_features,
        movie_features=movie_features,
        num_negatives=config.num_negatives,
    )

    model = DeepFM(
        field_dims=field_dims,
        embedding_dim=config.embedding_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    ).to(config.device)

    model = train(model, config, train_dataset, valid_dataset)

    for user in train_sequences:
        if user in valid_sequences:
            train_sequences[user] += valid_sequences[user]

    test_dataset = DeepFMEvalDataset(
        train_sequences=train_sequences,
        target_sequences=test_sequences,
        num_movies=num_movies,
        user_features=user_features,
        movie_features=movie_features,
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
