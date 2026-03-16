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


logger = logging.getLogger("model-ranking: DeepFM")


class MovieLensDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, ratings: pd.Series) -> None:
        self._data = torch.tensor(data.values, dtype=torch.long)
        self._ratings = torch.tensor(ratings.values, dtype=torch.float32)


    def __len__(self) -> int:
        return len(self._data)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._data[index], self._ratings[index]


class EmbeddingLayer(nn.Module):
    
    def __init__(self, feature_dimensions: Sequence[int], embedding_dimension: int):
        super(EmbeddingLayer, self).__init__()
        
        self._embeddings = nn.ModuleList([
            nn.Embedding(feature_dimension, embedding_dimension) for feature_dimension in feature_dimensions
        ])
        
        self._first_order_embeddings = nn.ModuleList([nn.Embedding(feature_dimension, 1) for feature_dimension in feature_dimensions])
        
        self._total_features = len(feature_dimensions)


    def forward(self, features_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_embeddings = [self._embeddings[i](features_batch[:, i]) for i in range(self._total_features)]
        first_order_embeddings = [self._first_order_embeddings[i](features_batch[:, i]) for i in range(self._total_features)]

        return torch.stack(first_order_embeddings, dim=1), torch.stack(feature_embeddings, dim=1)


class FactorizationMachineLayer(nn.Module):
    
    def __init__(self):
        super(FactorizationMachineLayer, self).__init__()
        self._bias = nn.Parameter(torch.zeros(1)) 


    def forward(self, first_order_embeddings: torch.Tensor, feature_embeddings: torch.Tensor) -> torch.Tensor:
        first_order_interactions = first_order_embeddings.sum(dim=1)      

        sum_then_square = (feature_embeddings.sum(dim=1) ** 2).sum(dim=1, keepdim=True)        
        square_then_sum = (feature_embeddings ** 2).sum(dim=[1, 2], keepdim=True).squeeze(-1)  
        second_order_interactions = 0.5 * (sum_then_square - square_then_sum)
 
        return first_order_interactions + second_order_interactions + self._bias


class DeepLayer(nn.Module):
 
    def __init__(
        self,
        num_features: int, 
        embedding_dimension: int,
        hidden_dims: Sequence = (400, 400, 400),
        dropout: float = 0.1 
    ):
        super(DeepLayer, self).__init__()
  
        layers = []

        prev_dim = num_features * embedding_dimension
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = hidden_dim
 
        layers.append(nn.Linear(prev_dim, 1, bias=False))
 
        self._mlp = nn.Sequential(*layers)
 
 
    def forward(self, feature_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = feature_embeddings.size(0)
        return self._mlp(feature_embeddings.view(batch_size, - 1))


class DeepFM(nn.Module):

    def __init__(
        self,
        feature_dimensions: Sequence[int],
        embedding_dimension: int,
        hidden_dims: Sequence = (400, 400, 400),
        dropout: float = 0.1,
    ):
        super(DeepFM, self).__init__()
 
        num_features = len(feature_dimensions)
    
        self._embedding_layer = EmbeddingLayer(feature_dimensions, embedding_dimension) 
        self._factorization_machine_layer = FactorizationMachineLayer() 
        self._deep_layer = DeepLayer(num_features, embedding_dimension, hidden_dims, dropout)


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        first_order_embeddings, feature_embeddings = self._embedding_layer(features)

        y_fml = self._factorization_machine_layer(first_order_embeddings, feature_embeddings)
        y_dl = self._deep_layer(feature_embeddings)       

        return y_fml + y_dl


def evaluate_model(model: nn.Module, data_loader: DataLoader, loss_function: nn.Module, device: torch.device) -> float:
    model.eval()
    
    total_loss = 0.0
    with torch.no_grad():
        for features_batch, ratings in data_loader:
            features_batch, ratings = features_batch.to(device), ratings.to(device)
            
            predictions = model(features_batch)
            
            total_loss += loss_function(predictions.squeeze(), ratings).item()

    model.train()

    return total_loss / len(data_loader)


def build_subsample_loader(dataset: Dataset, sample_size: int, batch_size: int) -> DataLoader:
    """
    Return a DataLoader over a random subset of dataset.
    """
    indices = torch.randperm(len(dataset))[:sample_size].tolist()

    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)


def train(
    model: nn.Module,
    loss_function: nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    epochs: int = 10,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    eval_batch_size: int = 512,
    train_eval_samples: int = 32_768,
    device: torch.device | None = None,
) -> nn.Module:

    if device is None:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

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

        for features_batch, ratings in train_loader:
            features_batch, ratings = features_batch.to(device), ratings.to(device)
            
            predictions = model(features_batch)
            
            loss = loss_function(predictions.squeeze(), ratings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_mse = evaluate_model(model, build_subsample_loader(train_dataset, train_eval_samples, eval_batch_size), loss_function, device)
        valid_mse = evaluate_model(model, valid_loader, loss_function, device)

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

    # NOTE: for simplicity we only handful of features
    feature_columns = ["user_id", "movie_id", "age", "occupation"]
    feature_dimensions = [int(data[col].max()) + 1 for col in feature_columns]

    train_dataset = MovieLensDataset(train_data[feature_columns], train_data["rating"])
    valid_dataset = MovieLensDataset(validation_data[feature_columns], validation_data["rating"])
    test_dataset  = MovieLensDataset(test_data[feature_columns],  test_data["rating"])

    model = DeepFM(
        feature_dimensions=feature_dimensions,
        embedding_dimension=16
    )

    loss_function = nn.MSELoss()

    trained_model = train(
        model=model,
        loss_function=loss_function,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=20
    )

    # Final evaluation on held-out test set
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    test_mse = evaluate_model(model, test_loader, loss_function, device)

    logger.info("Test MSE: %.4f | Test RMSE: %.4f", test_mse, test_mse ** 0.5)
