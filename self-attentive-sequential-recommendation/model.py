import torch
import torch.nn.functional as F

from torch import nn


class CausalSelfAttention(nn.Module):
 
    def __init__(self, embedding_dim: int, dropout: float):
        super(CausalSelfAttention, self).__init__()

        self._W_Q = nn.Linear(embedding_dim, embedding_dim)
        self._W_K = nn.Linear(embedding_dim, embedding_dim)
        self._W_V = nn.Linear(embedding_dim, embedding_dim)
        
        self._dropout = nn.Dropout(dropout)
        self._scale = embedding_dim ** 0.5
 

    def forward(self, sequence_embeddings_batch: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        Q = self._W_Q(sequence_embeddings_batch)
        K = self._W_K(sequence_embeddings_batch) 
        V = self._W_V(sequence_embeddings_batch)
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self._scale  # (batch_size, sequence_length, sequence_length)

        sequence_length = sequence_embeddings_batch.size(1)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=sequence_embeddings_batch.device, dtype=torch.bool), 
            diagonal=1
        )

        attention_scores = attention_scores.masked_fill(causal_mask, -1e9)
        attention_scores = attention_scores.masked_fill(padding_mask[:, None, :], -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(padding_mask[:, :, None], 0.0)

        attention_weights = self._dropout(attention_weights)
        
        return torch.bmm(attention_weights, V)  # (batch_size, sequence_length, embeddings_dim)


class FeedForward(nn.Module):
 
    def __init__(self, embedding_dim: int, dropout: float):
        super(FeedForward, self).__init__()

        self._feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )


    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self._feed_forward(embeddings)
    

class SelfAttentionBlock(nn.Module):
 
    def __init__(self, embedding_dim: int, dropout: float):
        super(SelfAttentionBlock, self).__init__()

        self._layer_norm_1 = nn.LayerNorm(embedding_dim)
        self._causal_self_attention = CausalSelfAttention(embedding_dim, dropout)

        self._layer_norm_2 = nn.LayerNorm(embedding_dim)
        self._ffn = FeedForward(embedding_dim, dropout)
        
        self._dropout = nn.Dropout(dropout)
 

    def forward(self, sequence_embeddings_batch: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # Sub-layer 1: self-attention
        sequence_embeddings_batch = sequence_embeddings_batch + self._dropout(
            self._causal_self_attention(self._layer_norm_1(sequence_embeddings_batch), padding_mask)
        )

        # Sub-layer 2: feed-forward
        sequence_embeddings_batch = sequence_embeddings_batch + self._dropout(self._ffn(self._layer_norm_2(sequence_embeddings_batch)))

        return sequence_embeddings_batch


class SelfAttentiveSequentialRecommender(nn.Module):
 
    def __init__(
        self, 
        num_items: int,
        embedding_dim: int = 16,
        context_length: int = 32,
        num_self_attention_blocks: int = 2,
        dropout: float = 0.05
    ) -> None:
        super(SelfAttentiveSequentialRecommender, self).__init__()

        self._context_length = context_length
        self._embedding_dim = embedding_dim
 
        self._item_embeddings = nn.Embedding(num_items + 1, self._embedding_dim, padding_idx=0)
        self._positional_embeddings = nn.Embedding(self._context_length, self._embedding_dim)
        self._embeddings_dropout = nn.Dropout(dropout)
 
        self._self_attention_blocks = nn.ModuleList([
            SelfAttentionBlock(self._embedding_dim, dropout) for _ in range(num_self_attention_blocks)
        ])
 
        self._final_norm = nn.LayerNorm(self._embedding_dim)
 

    def forward(self, item_sequence_batch: torch.Tensor) -> torch.Tensor:
        _, context_length = item_sequence_batch.shape
 
        item_embeddings = self._item_embeddings(item_sequence_batch)

        positions = torch.arange(context_length, device=item_sequence_batch.device).unsqueeze(0)
        positional_embeddings = self._positional_embeddings(positions) 
 
        sequence_embeddings_batch = self._embeddings_dropout(item_embeddings + positional_embeddings)
 
        padding_mask = item_sequence_batch == 0
        sequence_embeddings_batch = sequence_embeddings_batch.masked_fill(padding_mask[:, :, None], 0.0)
 
        for self_attention_block in self._self_attention_blocks:
            sequence_embeddings_batch = self_attention_block(sequence_embeddings_batch, padding_mask)
            sequence_embeddings_batch = sequence_embeddings_batch.masked_fill(padding_mask[:, :, None], 0.0)

        return self._final_norm(sequence_embeddings_batch)


    def predict_logits(self, item_sequence_batch: torch.Tensor) -> torch.Tensor:
        context_vectors = self.forward(item_sequence_batch)
        context_vectors = context_vectors[:, -1, :]

        logits = context_vectors @ self._item_embeddings.weight.T
        logits[:, 0] = float('-inf')

        return logits
