import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import logging

from typing import Sequence

import torch
import config
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from utils.movies import get_data, get_train_val_test_split


logger = logging.getLogger("model-ranking: NeuralCollaborativeFiltering")


class MovieLensDataset(Dataset):
    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data.values


    def __len__(self) -> int:
        return len(self._data)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_id  = torch.tensor(self._data[index, 0], dtype=torch.int)
        movie_id = torch.tensor(self._data[index, 1], dtype=torch.int)
        rating   = torch.tensor(self._data[index, 2], dtype=torch.float)

        return user_id, movie_id, rating


class NeuralCollaborativeFiltering(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_embedding_dim: int = 32,
        mlp_embedding_dim: int = 64,
        mlp_hidden_dims: Sequence[int] = (128, 32, 16, 8),
        dropout: float = 0.1
    ) -> None:
        super(NeuralCollaborativeFiltering, self).__init__()

        # GMF branch embeddings
        self._gmf_user_embedding = nn.Embedding(num_users, gmf_embedding_dim)
        self._gmf_item_embedding = nn.Embedding(num_items, gmf_embedding_dim)

        # MLP branch embeddings
        self._mlp_user_embedding = nn.Embedding(num_users, mlp_embedding_dim)
        self._mlp_item_embedding = nn.Embedding(num_items, mlp_embedding_dim)

        self._user_bias = nn.Embedding(num_users, 1)
        self._item_bias = nn.Embedding(num_items, 1)

        # MLP tower
        mlp_layers = []
        input_dim = 2 * mlp_embedding_dim

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self._mlp = nn.Sequential(*mlp_layers)

        # Final fusion layer
        final_input_dim = gmf_embedding_dim + mlp_hidden_dims[-1]
        self._output_layer = nn.Linear(final_input_dim, 1)


    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF branch
        gmf_user = self._gmf_user_embedding(user_ids)   # [B, gmf_dim]
        gmf_item = self._gmf_item_embedding(item_ids)   # [B, gmf_dim]
        gmf_out = gmf_user * gmf_item                   # [B, gmf_dim]

        # MLP branch
        mlp_user = self._mlp_user_embedding(user_ids)      # [B, mlp_dim]
        mlp_item = self._mlp_item_embedding(item_ids)      # [B, mlp_dim]
        mlp_in = torch.cat([mlp_user, mlp_item], dim=-1)   # [B, 2 * mlp_dim]
        mlp_out = self._mlp(mlp_in)                        # [B, last_hidden]

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)        # [B, gmf_dim + last_hidden]
        predictions = self._output_layer(fused).squeeze(-1)  # [B]

        user_bias = self._user_bias(user_ids).squeeze(-1)    # [B]
        item_bias = self._item_bias(item_ids).squeeze(-1)    # [B]

        return predictions + user_bias + item_bias


def evaluate_model(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    """
    Compute mean MSE loss over an entire DataLoader.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for wide, deep, ratings in data_loader:
            wide, deep, ratings = wide.to(device), deep.to(device), ratings.to(device)
            preds = model(wide, deep)
            total_loss += loss_fn(preds.squeeze(), ratings).item()

    model.train()

    return total_loss / len(data_loader)


def build_subsample_loader(dataset: Dataset, sample_size: int, batch_size: int) -> DataLoader:
    """
    Return a DataLoader over a random subset of *dataset*.
    """
    indices = torch.randperm(len(dataset))[:sample_size].tolist()
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)


def train(
    model: nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 2048,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_eval_samples: int = 32_768,
    eval_batch_size: int = 512,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False)

    logger.info("Starting training on %s", device)
    logger.info(
        "Train samples: %d | Validation samples: %d | Epochs: %d | Batch size: %d",
        len(train_dataset), len(valid_dataset), epochs, batch_size,
    )
    logger.info("-" * 90)

    for epoch in range(1, epochs + 1):
        model.train()

        for user_ids, movie_ids, ratings in train_loader:
            user_ids, movie_ids, ratings = user_ids.to(device), movie_ids.to(device), ratings.to(device)
            
            preds = model(user_ids, movie_ids)
            
            loss = loss_fn(preds.squeeze(), ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_mse = evaluate_model(model, build_subsample_loader(train_dataset, train_eval_samples, eval_batch_size), loss_fn, device)
        valid_mse = evaluate_model(model, valid_loader, loss_fn, device)

        logger.info(
            "Epoch %3d/%d | Train MSE: %.4f (RMSE: %.4f) | Validation MSE: %.4f (RMSE: %.4f)",
            epoch, epochs,
            train_mse, train_mse ** 0.5,
            valid_mse, valid_mse ** 0.5,
        )

    logger.info("-" * 90)
    
    return model


if __name__ == "__main__":
    data = get_data()
    train_data, validation_data, test_data = get_train_val_test_split(data)

    train_dataset = MovieLensDataset(train_data[["user_id", "movie_id", "rating"]])
    valid_dataset = MovieLensDataset(validation_data[["user_id", "movie_id", "rating"]])
    test_dataset = MovieLensDataset(test_data[["user_id", "movie_id", "rating"]])

    num_users = len(data.user_id.unique())
    num_items = len(data.movie_id.unique())

    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        gmf_embedding_dim=32,
        mlp_embedding_dim=32,
        mlp_hidden_dims=(64, 32, 16, 8),
        dropout=0.05
    )

    model = train(model,train_dataset, valid_dataset, epochs=20, batch_size=2048, lr=1e-3)

    # Final evaluation on held-out test set
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    test_mse = evaluate_model(model, test_loader, nn.MSELoss(), device)

    logger.info("Test MSE: %.4f | Test RMSE: %.4f", test_mse, test_mse ** 0.5)
