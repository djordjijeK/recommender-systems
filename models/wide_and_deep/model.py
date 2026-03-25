import torch

from torch import nn


class WideComponent(nn.Module):

    def __init__(self, wide_dim: int) -> None:
        super(WideComponent, self).__init__()

        self._linear = nn.Linear(wide_dim, 1)

    def forward(self, wide_features: torch.Tensor) -> torch.Tensor:
        return self._linear(wide_features).squeeze(-1)


class DeepComponent(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dimension: int = 256,
        hidden_layers: tuple[int, ...] = (1024, 512, 128, 32),
        dropout: float = 0.2,
    ) -> None:
        super(DeepComponent, self).__init__()

        self._user_embedding = nn.Embedding(num_users + 1, embedding_dimension, padding_idx=0)
        self._movie_embedding = nn.Embedding(num_movies + 1, embedding_dimension, padding_idx=0)

        layers: list[nn.Module] = []
        in_dim = 2 * embedding_dimension

        for hidden_dim in hidden_layers:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))

        self._mlp = nn.Sequential(*layers)


    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_embeddings = self._user_embedding(user_ids)
        movie_embeddings = self._movie_embedding(movie_ids)

        x = torch.cat([user_embeddings, movie_embeddings], dim=-1)
        return self._mlp(x).squeeze(-1)


class WideDeep(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        wide_dim: int,
        embedding_dimension: int = 256,
        hidden_layers: tuple[int, ...] = (1024, 512, 128, 32),
        dropout: float = 0.2,
    ) -> None:
        super(WideDeep, self).__init__()

        self._wide = WideComponent(wide_dim)
        self._deep = DeepComponent(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dimension=embedding_dimension,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )


    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        wide_features: torch.Tensor,
    ) -> torch.Tensor:
        return self._wide(wide_features) + self._deep(user_ids, movie_ids)


    def score_candidates(
        self,
        user_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_wide_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates = candidate_ids.shape

        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        candidate_ids_flat = candidate_ids.reshape(-1)
        candidate_wide_features_flat = candidate_wide_features.reshape(-1, candidate_wide_features.shape[-1])

        scores = self.forward(user_ids_expanded, candidate_ids_flat, candidate_wide_features_flat)

        return scores.reshape(batch_size, num_candidates)
