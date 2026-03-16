import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import logging

import torch
import config
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import SelfAttentiveSequentialRecommender
from dataset import build_dataset, SelfAttentiveSequentialRecommenderTrainDataset, SelfAttentiveSequentialRecommenderEvalDataset


logger = logging.getLogger("model:self-attentive-sequential-recommender")


class SelfAttentiveSequentialRecommenderLoss(nn.Module):

    def __init__(self):
        super(SelfAttentiveSequentialRecommenderLoss, self).__init__()


    def forward(
        self,
        sequence_output: torch.Tensor,              # (batch_size, sequence_length, embedding_dim)
        positive_target_ids: torch.Tensor,          # (batch_size, sequence_length)
        negative_target_ids: torch.Tensor,          # (batch_size, sequence_length)
        item_embedding_table: nn.Embedding,         # model._item_embeddings
    ) -> torch.Tensor:
        # Look up embeddings for positive and negative targets
        positive_emb = item_embedding_table(positive_target_ids)   # (batch_size, sequence_length, embedding_dim)
        negative_emb = item_embedding_table(negative_target_ids)   # (batch_size, sequence_length, embedding_dim)

        # Dot products between context vectors and target item embeddings
        positive_logits = (sequence_output * positive_emb).sum(dim=-1)  # (batch_size, sequence_length)
        negative_logits = (sequence_output * negative_emb).sum(dim=-1)  # (batch_size, sequence_length)

        valid_mask = positive_target_ids != 0
        if not valid_mask.any():
            return sequence_output.sum() * 0.0

        logits = torch.cat([
            positive_logits[valid_mask],
            negative_logits[valid_mask]
        ], dim=0)

        labels = torch.cat([
            torch.ones_like(positive_logits[valid_mask]),
            torch.zeros_like(negative_logits[valid_mask])
        ], dim=0)

        return F.binary_cross_entropy_with_logits(logits, labels)
    

def train(
    model: nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    epochs: int = 25,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    eval_batch_size: int = 512,
    k: int = 10,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    loss_function = SelfAttentiveSequentialRecommenderLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Training Self-Attentive Sequential Recommender on {device}")
    logger.info("-" * 85)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            input_sequence = batch["input_sequence"].to(device)                         # (batch_size, sequence_length)
            positive_targets_sequence = batch["positive_targets_sequence"].to(device)   # (batch_size, sequence_length)
            negative_targets_sequence = batch["negative_targets_sequence"].to(device)   # (batch_size, sequence_length)

            sequence_output = model(input_sequence)  # (batch_size, sequence_length, embedding_dim)

            loss = loss_function(
                sequence_output=sequence_output,
                positive_target_ids=positive_targets_sequence,
                negative_target_ids=negative_targets_sequence,
                item_embedding_table=model._item_embeddings,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_actual = input_sequence.size(0)
            running_loss += loss.item() * batch_size_actual
            total_examples += batch_size_actual

        train_loss = running_loss / max(total_examples, 1)

        valid_metrics = evaluate_model(
            model=model,
            eval_dataset=valid_dataset,
            batch_size=eval_batch_size,
            k=k,
            device=device,
        )

        valid_hit = valid_metrics[f"Hit@{k}"]
        valid_ndcg = valid_metrics[f"NDCG@{k}"]

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Hit@{k}: {valid_hit:.4f} | "
            f"Valid NDCG@{k}: {valid_ndcg:.4f}"
        )

    logger.info("-" * 85)

    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    eval_dataset: Dataset,
    batch_size: int = 512,
    k: int = 10,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    total_hit = 0.0
    total_ndcg = 0.0
    total_users = 0

    for batch in eval_loader:
        input_sequence = batch["input_sequence"].to(device)                       # (batch_size, sequence_length)
        positive_targets_sequence = batch["positive_targets_sequence"].to(device) # (batch_size, positive_items)
        negative_targets_sequence = batch["negative_targets_sequence"].to(device) # (batch_size, negative_items)

        target_item = positive_targets_sequence[:, -1]

        context_vectors = model(input_sequence)
        final_context = context_vectors[:, -1, :] 

        candidate_ids = torch.cat(
            [target_item.unsqueeze(1), negative_targets_sequence], dim=1
        )

        candidate_embeddings = model._item_embeddings(candidate_ids)
        scores = torch.bmm(candidate_embeddings, final_context.unsqueeze(-1)).squeeze(-1)

        batch_hit, batch_ndcg = hit_ndcg_at_k_from_scores(scores, k=k)

        batch_size_actual = input_sequence.size(0)

        total_hit += batch_hit * batch_size_actual
        total_ndcg += batch_ndcg * batch_size_actual
        total_users += batch_size_actual

    return {
        f"Hit@{k}": total_hit / total_users,
        f"NDCG@{k}": total_ndcg / total_users,
    }


@torch.no_grad()
def hit_ndcg_at_k_from_scores(scores: torch.Tensor, k: int = 10) -> tuple[float, float]:
    ground_truth_scores = scores[:, 0:1]
    ranks = 1 + (scores[:, 1:] > ground_truth_scores).sum(dim=1)

    hits = (ranks <= k).float()
    ndcgs = torch.where(
        ranks <= k,
        1.0 / torch.log2(ranks.float() + 1.0),
        torch.zeros_like(ranks, dtype=torch.float),
    )

    return hits.mean().item(), ndcgs.mean().item()


if __name__ == "__main__":
    context_length = 128
    num_negatives = 64
    embedding_dim = 64
    num_self_attention_blocks = 4
    dropout = 0.15
    k = 10
    epochs = 30

    train_sequences, valid_sequences, test_sequences, num_users, num_items = build_dataset()

    train_dataset = SelfAttentiveSequentialRecommenderTrainDataset(
        train_sequences=train_sequences,
        num_items=num_items,
        context_length=context_length
    )

    valid_dataset = SelfAttentiveSequentialRecommenderEvalDataset(
        train_sequences = train_sequences, 
        target_sequences = valid_sequences,
        num_items = num_items, 
        context_length = context_length,
        num_negatives = num_negatives
    )

    test_dataset = SelfAttentiveSequentialRecommenderEvalDataset(
        train_sequences = train_sequences, 
        target_sequences = test_sequences,
        num_items = num_items, 
        context_length = context_length,
        num_negatives = num_negatives
    )

    model = SelfAttentiveSequentialRecommender(
        num_items=num_items,
        embedding_dim=embedding_dim,
        context_length=context_length,
        num_self_attention_blocks=num_self_attention_blocks,
        dropout=dropout
    )

    model = train(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=epochs,
        batch_size=128,
        learning_rate=1e-3,
        eval_batch_size=512,
        k=k
    )

    test_metrics = evaluate_model(
        model=model,
        eval_dataset=test_dataset
    )

    logger.info(
        f"Test Hit@{k}: {test_metrics[f"Hit@{k}"]:.4f} | "
        f"Test NDCG@{k}: {test_metrics[f"NDCG@{k}"]:.4f}"
    )    
