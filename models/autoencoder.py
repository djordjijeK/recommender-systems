import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import torch
import logging

from torch import nn
from typing import Tuple
from torch.utils.data import DataLoader
from utils.datav1 import get_user_item_interaction_data, leave_k_out_split, get_device, UserItemMatrix


logger = logging.getLogger("model:autoencoder")


class Autoencoder(nn.Module):
    
    def __init__(self, num_items: int, hidden_dim: int, dropout: float = 0.05):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Linear(num_items, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, num_items, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.encoder(x))
        hidden = self.dropout(hidden)
        out = self.decoder(hidden)

        return out


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, training_dataset: UserItemMatrix | None = None) -> Tuple[float, float]:
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_observed = 0

    with torch.no_grad():
        for indexes, user_ratings, user_ratings_mask in data_loader:
            if training_dataset is not None:
                ratings = training_dataset._user_item_matrix[indexes].to(device)
            else:
                ratings = user_ratings.to(device)

            user_ratings = user_ratings.to(device)
            user_ratings_mask = user_ratings_mask.to(device)

            predictions = model(ratings)
            observed_predictions = predictions[user_ratings_mask == 1]
            observed_targets = user_ratings[user_ratings_mask == 1]

            total_mse += (observed_predictions - observed_targets).pow(2).sum().item()
            total_mae += (observed_predictions - observed_targets).abs().sum().item()
            total_observed += observed_targets.numel()

        n = max(total_observed, 1)
        rmse = (total_mse / n) ** 0.5
        mae = total_mae / n

    return rmse, mae 


if __name__ == "__main__":
    # 1. Load data
    data_frame = get_user_item_interaction_data()

    num_users = len(data_frame.user_id.unique())
    num_items = len(data_frame.item_id.unique())

    training_df, test_df = leave_k_out_split(data_frame, random_state=1347)

    training_dataset = UserItemMatrix(training_df, num_users, num_items)
    test_dataset = UserItemMatrix(test_df, num_users, num_items)

    training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=256)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=512)

    # 2. Train & evaluate model
    device = get_device(force_cpu_execution=True)

    model = Autoencoder(num_items=num_items, hidden_dim=64)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()

        for _, user_ratings, user_ratings_mask in training_data_loader:
            user_ratings = user_ratings.to(device)
            user_ratings_mask = user_ratings_mask.to(device)

            optimizer.zero_grad()

            predictions = model(user_ratings)
            observed_predictions = predictions[user_ratings_mask == 1]
            observed_targets = user_ratings[user_ratings_mask == 1]

            loss = loss_fn(observed_predictions, observed_targets)
            loss.backward()

            optimizer.step()

        if (epoch + 1) % 10 == 0:
            train_rmse, train_mae = evaluate(model, training_data_loader, device)
            test_rmse, test_mae = evaluate(model, test_data_loader, device, training_dataset)

            logger.info(f"Epoch {epoch + 1:3d}/{n_epochs:3d} | {'Train RMSE':10s}: {train_rmse:.4f} | {'Test RMSE':9s}: {test_rmse:.4f}")
            logger.info(f"Epoch {epoch + 1:3d}/{n_epochs:3d} | {'Train MAE':10s}: {train_mae:.4f} | {'Test MAE':9s}: {test_mae:.4f}")
