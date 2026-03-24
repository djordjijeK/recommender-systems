import math
import torch
import torch.nn.functional as F

from torch import nn


class CausalSelfAttention(nn.Module):

    def __init__(self, n_dim: int, dropout: float) -> None:
        super(CausalSelfAttention, self).__init__()

        self._W_Q = nn.Linear(n_dim, n_dim)
        self._W_K = nn.Linear(n_dim, n_dim)
        self._W_V = nn.Linear(n_dim, n_dim)

        self._dropout = nn.Dropout(dropout)
        self._scale = math.sqrt(n_dim)


    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        Q = self._W_Q(x)
        K = self._W_K(x)
        V = self._W_V(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self._scale  # (batch_size, context_length, context_length)

        context_length = x.size(1)
        causal_mask = torch.triu(
            torch.ones(context_length, context_length, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(causal_mask, float("-inf"))
        scores = scores.masked_fill(padding_mask[:, None, :], float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        attention_weights = attention_weights.masked_fill(padding_mask[:, :, None], 0.0)

        attention_weights = self._dropout(attention_weights)

        return torch.bmm(attention_weights, V)  # (batch_size, context_length, n_dim)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, n_dim: int, dropout: float) -> None:
        super(PositionWiseFeedForward, self).__init__()

        self._W1 = nn.Linear(n_dim, n_dim)
        self._W2 = nn.Linear(n_dim, n_dim)
        self._dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dropout(self._W2(F.relu(self._W1(x))))


class Transformer(nn.Module):

    def __init__(self, n_dim: int, dropout: float) -> None:
        super(Transformer, self).__init__()

        self._norm1 = nn.LayerNorm(n_dim)
        self._causal_self_attention = CausalSelfAttention(n_dim, dropout)

        self._norm2 = nn.LayerNorm(n_dim)
        self._ffn = PositionWiseFeedForward(n_dim, dropout)

        self._dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self._dropout(self._causal_self_attention(self._norm1(x), padding_mask))
        x = x + self._dropout(self._ffn(self._norm2(x)))

        return x


class SASRec(nn.Module):

    def __init__(
        self,
        num_movies: int,
        context_length: int = 200,
        n_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(SASRec, self).__init__()

        self._num_movies = num_movies
        self._pad_token = 0

        vocab_size = num_movies + 1

        self.context_length = context_length
        self._n_dim = n_dim

        self._movie_embedding = nn.Embedding(vocab_size, n_dim, padding_idx=self._pad_token)
        self._positional_embedding = nn.Embedding(context_length, n_dim)

        self._embedding_dropout = nn.Dropout(dropout)

        self._transformer_layers = nn.ModuleList([
            Transformer(n_dim, dropout) for _ in range(n_layers)
        ])

        self._final_norm = nn.LayerNorm(n_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, context_length = tokens.shape

        positions = torch.arange(context_length, device=tokens.device).unsqueeze(0)
        embeddings = self._movie_embedding(tokens) + self._positional_embedding(positions)
        embeddings = self._embedding_dropout(embeddings)

        padding_mask = tokens == self._pad_token
        embeddings = embeddings.masked_fill(padding_mask[:, :, None], 0.0)

        for transformer_layer in self._transformer_layers:
            embeddings = transformer_layer(embeddings, padding_mask)
            embeddings = embeddings.masked_fill(padding_mask[:, :, None], 0.0)

        return self._final_norm(embeddings)  # (batch_size, context_length, n_dim)


    def predict_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        context_vectors = self.forward(tokens)
        last_context = context_vectors[:, -1, :]  # (batch_size, n_dim)

        logits = last_context @ self._movie_embedding.weight.T  # (batch_size, vocab_size)
        logits[:, self._pad_token] = float("-inf")

        return logits


    @torch.no_grad()
    def recommend(
        self,
        tokens: torch.Tensor,
        k: int = 10
    ) -> torch.Tensor:
        logits = self.predict_logits(tokens)  # (batch_size, vocab_size)

        return torch.topk(logits, k, dim=-1).indices  # (batch_size, k)
