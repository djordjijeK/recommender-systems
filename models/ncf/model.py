import torch

from torch import nn
from typing import Sequence


class GMF(nn.Module):

    def __init__(self, num_users: int, num_movies: int, embedding_dim: int) -> None:
        super(GMF, self).__init__()

        self._user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self._item_embedding = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)


    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self._user_embedding(user_ids)     # [B, gmf_dim]
        item_emb = self._item_embedding(movie_ids)    # [B, gmf_dim]

        return user_emb * item_emb                     # [B, gmf_dim]


class MLP(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embedding_dim: int,
        hidden_layers: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super(MLP, self).__init__()

        self._user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self._item_embedding = nn.Embedding(num_movies + 1, embedding_dim, padding_idx=0)

        layers: list[nn.Module] = []
        in_dim = 2 * embedding_dim

        for hidden_dim in hidden_layers:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim

        self._mlp = nn.Sequential(*layers)
        self.output_dim = hidden_layers[-1]


    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        user_emb = self._user_embedding(user_ids)       # [B, mlp_dim]
        item_emb = self._item_embedding(movie_ids)      # [B, mlp_dim]

        x = torch.cat([user_emb, item_emb], dim=-1)     # [B, 2 * mlp_dim]
        return self._mlp(x)                             # [B, last_hidden]


class NeuMF(nn.Module):

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        gmf_embedding_dim: int = 64,
        mlp_embedding_dim: int = 128,
        mlp_hidden_layers: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super(NeuMF, self).__init__()

        self._gmf = GMF(num_users, num_movies, gmf_embedding_dim)
        self._mlp = MLP(num_users, num_movies, mlp_embedding_dim, mlp_hidden_layers, dropout)

        self._output_layer = nn.Linear(gmf_embedding_dim + self._mlp.output_dim, 1)


    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> torch.Tensor:
        gmf_out = self._gmf(user_ids, movie_ids)    # [B, gmf_dim]
        mlp_out = self._mlp(user_ids, movie_ids)    # [B, last_hidden]

        fused = torch.cat([gmf_out, mlp_out], dim=-1)  # [B, gmf_dim + last_hidden]
        return self._output_layer(fused).squeeze(-1)   # [B]


    def score_candidates(
        self,
        user_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates = candidate_ids.shape

        user_ids_expanded = user_ids.unsqueeze(1).expand(-1, num_candidates).reshape(-1)
        candidate_ids_flat = candidate_ids.reshape(-1)

        scores = self.forward(user_ids_expanded, candidate_ids_flat)

        return scores.reshape(batch_size, num_candidates)
