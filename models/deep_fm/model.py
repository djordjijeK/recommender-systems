import torch

from torch import nn
from typing import Sequence


class DeepFM(nn.Module):

    def __init__(
        self,
        field_dims: Sequence[int],
        embedding_dim: int = 64,
        hidden_layers: Sequence[int] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        super(DeepFM, self).__init__()

        num_fields = len(field_dims)

        self._embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in field_dims
        ])

        self._linear_weights = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])

        self._bias = nn.Parameter(torch.zeros(1))

        layers: list[nn.Module] = []
        in_dim = num_fields * embedding_dim

        for hidden_dim in hidden_layers:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self._mlp = nn.Sequential(*layers)


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latent_vectors = torch.stack([
            emb(features[:, i]) for i, emb in enumerate(self._embeddings)
        ], dim=1)

        linear_terms = torch.stack([
            emb(features[:, i]) for i, emb in enumerate(self._linear_weights)
        ], dim=1)

        # FM: bias + first-order + second-order
        first_order = linear_terms.sum(dim=1).squeeze(-1)  

        sum_of_vectors = latent_vectors.sum(dim=1)                         
        square_of_sum = sum_of_vectors.pow(2).sum(dim=-1)                   
        sum_of_squares = latent_vectors.pow(2).sum(dim=[1, 2])              
        second_order = 0.5 * (square_of_sum - sum_of_squares)              

        y_fm = self._bias.squeeze() + first_order + second_order            

        # Deep: MLP on flattened shared embeddings
        batch_size = features.size(0)
        y_deep = self._mlp(latent_vectors.reshape(batch_size, -1)).squeeze(-1)  

        return y_fm + y_deep 


    def score_candidates(self, candidate_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_candidates, num_fields = candidate_features.shape

        scores = self.forward(candidate_features.reshape(batch_size * num_candidates, num_fields))

        return scores.reshape(batch_size, num_candidates)

