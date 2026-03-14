import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import torch
import logging
import pandas as pd

from torch import nn
from typing import Any, Tuple
from torch.utils.data import DataLoader
from utils.datav1 import get_user_item_interaction_data, leave_k_out_split, get_device, UserItemInteractions


logger = logging.getLogger("model:matrix-factorization")


class MatrixFactorization(nn.Module):

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super(MatrixFactorization, self).__init__()

        self.__user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.__item_embeddings = nn.Embedding(n_items, embedding_dim)

        self.__user_bias = nn.Embedding(n_users, 1)
        self.__item_bias = nn.Embedding(n_items, 1)

    
    def forward(self, user_id_batch, item_id_batch):
        user_embeddings = self.__user_embeddings(user_id_batch - 1)
        item_embeddings = self.__item_embeddings(item_id_batch - 1)

        user_biases = self.__user_bias(user_id_batch - 1)
        item_biases = self.__item_bias(item_id_batch - 1)

        return (user_embeddings * item_embeddings).sum(axis = 1) + user_biases.squeeze(-1) + item_biases.squeeze(-1)


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()

    squared_errors = 0.0
    absolute_errors = 0.0
    total_evaluations = 0

    with torch.no_grad():
        for user_id_batch, item_id_batch, ratings in data_loader:
            predictions = model(user_id_batch.to(device), item_id_batch.to(device))
            errors = predictions - ratings.to(torch.float32).to(device)
            
            squared_errors += errors.pow(2).sum().item()
            absolute_errors += errors.abs().sum().item()

            total_evaluations += errors.numel()

    mse = squared_errors / total_evaluations
    
    return mse ** 0.5, absolute_errors / total_evaluations 


if __name__ == "__main__":
    # 1. Load data
    data_frame = get_user_item_interaction_data()

    num_users = len(data_frame.user_id.unique())
    num_items = len(data_frame.item_id.unique())

    data_frame_train, data_frame_test = leave_k_out_split(data_frame, random_state=1347)

    training_dataset = UserItemInteractions(data_frame_train)
    test_dataset = UserItemInteractions(data_frame_test)

    training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=256)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=512)

    # 2. Train & evaluate model
    device = get_device(force_cpu_execution=True)

    model = MatrixFactorization(n_users=num_users, n_items=num_items, embedding_dim=64)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()

        for user_id_batch, item_id_batch, ratings in training_data_loader:
            user_id_batch = user_id_batch.to(device)
            item_id_batch = item_id_batch.to(device)
            ratings = ratings.to(torch.float32).to(device)

            optimizer.zero_grad()
            predictions = model(user_id_batch, item_id_batch)
            loss = loss_fn(predictions, ratings)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            train_rmse, train_mae = evaluate(model, training_data_loader, device)
            test_rmse, test_mae = evaluate(model, test_data_loader, device)

            logger.info(f"Epoch {epoch + 1:3d}/{n_epochs:3d} | {'Train RMSE':10s}: {train_rmse:.4f} | {'Test RMSE':9s}: {test_rmse:.4f}")
            logger.info(f"Epoch {epoch + 1:3d}/{n_epochs:3d} | {'Train MAE':10s}: {train_mae:.4f} | {'Test MAE':9s}: {test_mae:.4f}")
