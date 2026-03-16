import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))


import logging

import torch
import config
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from utils.movies import get_data, get_train_val_test_split, AGE_MAP, OCCUPATION_MAP


logger = logging.getLogger("model-ranking: Wide&Deep")


def get_wide_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Build the wide (memorisation) feature matrix.

    Produces one-hot encodings for age, gender and occupation, then adds
    cross-product interaction terms between gender × genre and occupation × genre.
    """
    df = data.copy()

    df["age"] = df["age"].map(AGE_MAP)
    df["gender"] = df["gender"].map({0: "male", 1: "female"})
    df["occupation"] = df["occupation"].map(OCCUPATION_MAP)

    age_dummies = pd.get_dummies(df["age"], prefix="age", dtype=int)
    gender_dummies = pd.get_dummies(df["gender"], prefix="gender", dtype=int)
    occupation_dummies = pd.get_dummies(df["occupation"], prefix="occupation", dtype=int)

    df = pd.concat([df, age_dummies, gender_dummies, occupation_dummies], axis=1)

    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    gender_cols = list(gender_dummies.columns)
    occupation_cols = list(occupation_dummies.columns)

    interactions: dict[str, pd.Series] = {}

    for gender_col in gender_cols:
        for genre_col in genre_cols:
            interactions[f"{gender_col}_{genre_col}"] = df[gender_col] * df[genre_col]

    for occupation_col in occupation_cols:
        for genre_col in genre_cols:
            interactions[f"{occupation_col}_{genre_col}"] = df[occupation_col] * df[genre_col]

    df = pd.concat([df, pd.DataFrame(interactions, index=df.index)], axis=1)
    df = df.drop(columns=["user_id", "movie_id", "rating", "timestamp", "gender", "age", "occupation", "release_year"])

    return df


def get_deep_features(data: pd.DataFrame, release_year_min: int, release_year_max: int) -> pd.DataFrame:
    """
    Build the deep (generalisation) feature matrix.

    Keeps categorical ID columns (user, movie, gender, age, occupation) as
    raw integer indices for embedding lookup, and normalises ``release_year``
    to [0, 1].
    """
    df = data.copy()

    df["release_year"] = (df["release_year"] - release_year_min) / (release_year_max - release_year_min)
    df = df.drop(columns=["rating", "timestamp"])

    return df


class MovieLensDataset(Dataset):
    """
    Dataset for the MovieLens Wide & Deep model.

    Each sample is a ``(wide_features, deep_features, rating)`` tuple where:

    * ``wide_features``: 1-D float tensor of one-hot and interaction terms.
    * ``deep_features``: 1-D float tensor whose first five columns are categorical IDs ``[user_id, movie_id, gender, age, occupation]`` followed by continuous features (genre flags and normalised release year).
    * ``rating``: scalar float tensor with the ground-truth star rating.
    """

    def __init__(self, wide_features: pd.DataFrame, deep_features: pd.DataFrame, ratings: pd.Series) -> None:
        if len(wide_features) != len(deep_features):
            raise ValueError(
                f"Wide and deep feature frames must have the same length, "
                f"got {len(wide_features)} vs {len(deep_features)}."
            )

        self._wide = wide_features.values.astype("float32")
        self._deep = deep_features.values.astype("float32")
        self._ratings = ratings.values.astype("float32")


    def __len__(self) -> int:
        return len(self._wide)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wide = torch.from_numpy(self._wide[index])
        deep = torch.from_numpy(self._deep[index])
        rating = torch.tensor(self._ratings[index])

        return wide, deep, rating


class WideComponent(nn.Module):
    """
    Linear model applied to the wide (hand-crafted) feature vector.

    A single fully-connected layer without a non-linearity. Captures
    feature memorisation via the cross-product interaction terms baked into
    the input.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self._linear = nn.Linear(in_features, out_features=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)


class DeepComponent(nn.Module):
    """
    MLP applied to learned embeddings of categorical features.

    Categorical variables (user, movie, gender, age, occupation) are looked
    up in separate embedding tables and concatenated with continuous features
    before being fed through a multi-layer perceptron.
    """

    def __init__(
        self,
        user_config: tuple[int, int],
        movie_config: tuple[int, int],
        gender_config: tuple[int, int],
        age_config: tuple[int, int],
        occupation_config: tuple[int, int],
        num_continuous_features: int,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self._user_embedding = nn.Embedding(*user_config)
        self._movie_embedding = nn.Embedding(*movie_config)
        self._gender_embedding = nn.Embedding(*gender_config)
        self._age_embedding = nn.Embedding(*age_config)
        self._occupation_embedding = nn.Embedding(*occupation_config)

        embedding_dim_total = sum(cfg[1] for cfg in [
            user_config, movie_config, gender_config, age_config, occupation_config
        ])

        mlp_input_dim = embedding_dim_total + num_continuous_features

        layers: list[nn.Module] = []
        in_dim = mlp_input_dim

        for hidden_dim in hidden_layers:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self._mlp = nn.Sequential(*layers)


    def forward(
        self,
        user_id: torch.Tensor,
        movie_id: torch.Tensor,
        gender: torch.Tensor,
        age: torch.Tensor,
        occupation: torch.Tensor,
        continuous_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([
            self._user_embedding(user_id),
            self._movie_embedding(movie_id),
            self._gender_embedding(gender),
            self._age_embedding(age),
            self._occupation_embedding(occupation),
            continuous_features,
        ], dim=1)

        return self._mlp(x)


class WideDeep(nn.Module):
    """
    Wide & Deep model for rating prediction.

    Combines a linear wide component (memorisation) with a deep embedding MLP
    (generalisation) by summing their scalar outputs.
    """

    def __init__(
        self,
        wide_in_features: int,
        user_config: tuple[int, int],
        movie_config: tuple[int, int],
        gender_config: tuple[int, int],
        age_config: tuple[int, int],
        occupation_config: tuple[int, int],
        num_continuous_features: int,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.wide = WideComponent(in_features=wide_in_features)
        self.deep = DeepComponent(
            user_config=user_config,
            movie_config=movie_config,
            gender_config=gender_config,
            age_config=age_config,
            occupation_config=occupation_config,
            num_continuous_features=num_continuous_features,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )


    def forward(
        self,
        wide_features: torch.Tensor,
        deep_features: torch.Tensor,
    ) -> torch.Tensor:
        wide_out = self.wide(wide_features)

        user_id = deep_features[:, 0].long()
        movie_id = deep_features[:, 1].long()
        gender = deep_features[:, 2].long()
        age = deep_features[:, 3].long()
        occupation = deep_features[:, 4].long()
        continuous = deep_features[:, 5:].float()

        deep_out = self.deep(
            user_id=user_id,
            movie_id=movie_id,
            gender=gender,
            age=age,
            occupation=occupation,
            continuous_features=continuous,
        )

        return wide_out + deep_out


def evaluate_model(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for wide_features, deep_features, ratings in data_loader:
            wide_features, deep_features, ratings = wide_features.to(device), deep_features.to(device), ratings.to(device)
            predictions = model(wide_features, deep_features)

            total_loss += loss_fn(predictions.squeeze(), ratings).item()

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
    loss_fn: nn.Module,
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

        for wide_features, deep_features, ratings in train_loader:
            wide_features, deep_features, ratings = wide_features.to(device), deep_features.to(device), ratings.to(device)
            predictions = model(wide_features, deep_features)

            loss = loss_fn(predictions.squeeze(), ratings)
            
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

    release_year_min = int(data["release_year"].min())
    release_year_max = int(data["release_year"].max())

    train_wide = get_wide_features(train_data)
    train_deep = get_deep_features(train_data, release_year_min, release_year_max)

    valid_wide = get_wide_features(validation_data)
    valid_deep = get_deep_features(validation_data, release_year_min, release_year_max)

    test_wide = get_wide_features(test_data)
    test_deep = get_deep_features(test_data, release_year_min, release_year_max)

    train_dataset = MovieLensDataset(train_wide, train_deep, train_data["rating"])
    valid_dataset = MovieLensDataset(valid_wide, valid_deep, validation_data["rating"])
    test_dataset = MovieLensDataset(test_wide, test_deep, test_data["rating"])

    model = WideDeep(
        wide_in_features=train_wide.shape[1],
        user_config=(int(data["user_id"].max()) + 1, 32),
        movie_config=(int(data["movie_id"].max()) + 1, 32),
        gender_config=(2, 4),
        age_config=(int(data["age"].max()) + 1, 16),
        occupation_config=(int(data["occupation"].max()) + 1, 16),
        num_continuous_features=train_deep.shape[1] - 5,
        dropout=0.1
    )

    loss_function = nn.MSELoss()

    model = train(model, loss_function, train_dataset, valid_dataset, epochs=20, batch_size=2048, learning_rate=1e-3)

    # Final evaluation on held-out test set
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    test_mse = evaluate_model(model, test_loader, nn.MSELoss(), device)

    logger.info("Test MSE: %.4f | Test RMSE: %.4f", test_mse, test_mse ** 0.5)