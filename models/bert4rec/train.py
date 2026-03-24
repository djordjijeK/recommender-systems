import math
import torch
import config
import logging
import torch.nn.functional as F

from dataclasses import dataclass
from .model import BERT4Rec
from torch.utils.data import DataLoader
from .data import build_sequences, BERT4RecTrainDataset, BERT4RecEvalDataset


logger = logging.getLogger("model:BERT4Rec")


@dataclass
class Config:
    min_interactions: int = 5
 
    context_length: int = 128
    n_dim: int = 128
    n_layers: int = 2       
    n_heads: int = 2
    dropout: float = 0.2
 
    mask_probability: float = 0.2
    force_last_item_mask_probability: float = 0.2
 
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 1e-3
 
    num_negatives: int = 100
 
    device: str = "mps" if torch.mps.is_available() else "cpu"


@torch.no_grad()
def evaluate(
    model:  BERT4Rec,
    data_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    all_ranks: list[torch.Tensor] = []
 
    for batch_data in data_loader:
        tokens = batch_data["tokens"].to(device)   
        positive = batch_data["positive_movie"].to(device)  
        negatives = batch_data["negative_movies"].to(device) 
 
        last_logit = model(tokens)[:, -1, :]
 
        positive_movie_score = last_logit.gather(1, positive.unsqueeze(1))
        negative_movies_scores = last_logit.gather(1, negatives)   
 
        rank = (negative_movies_scores >= positive_movie_score).sum(dim=1) + 1 
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
    model: BERT4Rec,
    config: Config,
    train_dataset: BERT4RecTrainDataset,
    valid_dataset: BERT4RecEvalDataset
) -> BERT4Rec: 
    device = torch.device(config.device)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size  = config.batch_size,
        shuffle = True
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size * 2,
        shuffle = False
    )
 
    optimizer = torch.optim.AdamW(params=model.parameters(), lr = config.learning_rate)
  
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)

    logger.info(f"Training BERT4Rec on {device}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("=" * 100)
 
    for epoch in range(1, config.num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        for train_batch in train_data_loader:
            tokens = train_batch["tokens"].to(device) 
            labels = train_batch["labels"].to(device)   

            logits = model(tokens)
            batch_size, context_length, vocab_size = logits.shape
 
            loss = F.cross_entropy(
                logits.view(batch_size * context_length, vocab_size),
                labels.view(batch_size * context_length),
                ignore_index=0
            )
 
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

    train_tokens, valid_tokens, test_tokens, num_users, num_movies = build_sequences(min_interactions=config.min_interactions)

    train_dataset = BERT4RecTrainDataset(
        train_tokens = train_tokens,
        num_movies = num_movies,
        context_length = config.context_length,
        mask_prob = config.mask_probability,
        force_last_item_mask_probability = config.force_last_item_mask_probability
    )

    valid_dataset = BERT4RecEvalDataset(
        train_tokens = train_tokens,
        target_tokens = valid_tokens, 
        num_movies = num_movies,
        context_length = config.context_length,
        num_negatives = config.num_negatives
    )

    model = BERT4Rec(
        num_movies = num_movies,
        context_length = config.context_length,
        n_dim = config.n_dim,
        n_layers = config.n_layers,
        n_heads = config.n_heads,
        dropout = config.dropout,
    ).to(config.device)

    model = train(model, config, train_dataset, valid_dataset)

    for user in train_tokens:
        train_tokens[user] += valid_tokens[user]

    test_dataset = BERT4RecEvalDataset(
        train_tokens = train_tokens,
        target_tokens = test_tokens, 
        num_movies = num_movies,
        context_length = config.context_length,
        num_negatives = config.num_negatives
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size * 2,
        shuffle = False
    )

    test_metrics = evaluate(model, test_data_loader, config.device)
    metrics = " | ".join(f"{k}: {v:.4f}" for k, v in test_metrics.items())

    logger.info(f"Test metrics - {metrics}")