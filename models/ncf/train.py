import math
import torch
import config
import logging
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader

from .model import NeuMF
from .data import build_sequences, NCFTrainDataset, NCFEvalDataset


logger = logging.getLogger("model:NCF")


@dataclass
class Config:
    min_interactions: int = 5

    gmf_embedding_dim: int = 32
    mlp_embedding_dim: int = 32
    mlp_hidden_layers: tuple[int, ...] = (128, 64, 32, 16)
    dropout: float = 0.0

    num_train_negatives: int = 4

    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 1e-3

    num_negatives: int = 100

    device: str = "mps" if torch.mps.is_available() else "cpu"


@torch.no_grad()
def evaluate(model: NeuMF, data_loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_ranks: list[torch.Tensor] = []

    for batch_data in data_loader:
        user_ids = batch_data["user_id"].to(device)
        candidate_ids = batch_data["candidate_ids"].to(device)

        scores = model.score_candidates(user_ids, candidate_ids)

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
    model: NeuMF,
    config: Config,
    train_dataset: NCFTrainDataset,
    valid_dataset: NCFEvalDataset,
) -> NeuMF:
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

    logger.info(f"Training NCF (NeuMF) on {device}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("=" * 100)

    for epoch in range(1, config.num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        for train_batch in train_data_loader:
            user_ids = train_batch["user_id"].to(device)                 # [B]
            positive_movie = train_batch["positive_movie"].to(device)    # [B]
            negative_movies = train_batch["negative_movies"].to(device)  # [B, num_negatives]

            batch_size = user_ids.size(0)
            num_negatives = negative_movies.size(1)

            positive_scores = model(user_ids, positive_movie)  # [B]

            user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
            negative_scores = model(user_ids_expanded, negative_movies.reshape(-1)).reshape(batch_size, num_negatives)  # [B, num_negatives]

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

    train_sequences, valid_sequences, test_sequences, num_users, num_movies = build_sequences(min_interactions=config.min_interactions)

    train_dataset = NCFTrainDataset(
        train_sequences=train_sequences,
        num_movies=num_movies,
        num_negatives=config.num_train_negatives,
    )

    valid_dataset = NCFEvalDataset(
        train_sequences=train_sequences,
        target_sequences=valid_sequences,
        num_movies=num_movies,
        num_negatives=config.num_negatives,
    )

    model = NeuMF(
        num_users=num_users,
        num_movies=num_movies,
        gmf_embedding_dim=config.gmf_embedding_dim,
        mlp_embedding_dim=config.mlp_embedding_dim,
        mlp_hidden_layers=config.mlp_hidden_layers,
        dropout=config.dropout,
    ).to(config.device)

    model = train(model, config, train_dataset, valid_dataset)

    for user in train_sequences:
        if user in valid_sequences:
            train_sequences[user] += valid_sequences[user]

    test_dataset = NCFEvalDataset(
        train_sequences=train_sequences,
        target_sequences=test_sequences,
        num_movies=num_movies,
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
