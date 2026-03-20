import math
import torch
import torch.nn.functional as F

from torch import nn


class MultiHeadSelfAttention(nn.Module):
 
    def __init__(self, n_dim: int, n_heads: int, dropout: float) -> None:
        super(MultiHeadSelfAttention, self).__init__()
 
        assert n_dim % n_heads == 0, "n_dim must be divisible by n_heads"
 
        self.n_heads = n_heads
        self.head_dim = n_dim // n_heads 
  
        self._W_Q = nn.Linear(n_dim, n_dim)
        self._W_K = nn.Linear(n_dim, n_dim)
        self._W_V = nn.Linear(n_dim, n_dim)
 
        self._W_O = nn.Linear(n_dim, n_dim)
 
        self._dropout = nn.Dropout(dropout)


    def __split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, context_size, n_dim = x.shape
        return x.view(batch_size, context_size, self.n_heads, self.head_dim).transpose(1, 2)


    def __merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, context_length, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, context_length, n_heads * head_dim)
    

    def __scaled_dot_product_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        # Q, K, V -> (batch_size, num_heads, context_length, head_dim)
    
        # attention scores (batch_size, num_heads, context_length, context_length)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    
        # zero out attention to PAD positions (set to -inf so softmax → 0)
        scores = scores.masked_fill(padding_mask, float("-inf"))
    
        attention_weights = F.softmax(scores, dim=-1)
    
        # replace NaN that can appear when entire row is -inf (fully-padded tokens)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
        return torch.matmul(attention_weights, V)  # (batch_size, num_heads, context_length, head_dim)


    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # Q, K, V -> (batch_size, num_heads, context_length, head_dim)
        # padding_mask -> (batch_size, 1, 1, context_length)
        Q = self.__split_heads(self._W_Q(x))
        K = self.__split_heads(self._W_K(x))
        V = self.__split_heads(self._W_V(x))
 
        context_vectors = self.__scaled_dot_product_attention(Q, K, V, padding_mask)
        context_vectors = self.__merge_heads(context_vectors)  # (batch_size, context_length, n_dim)
 
        context_vectors = self._W_O(context_vectors)
        return self._dropout(context_vectors)


class PositionWiseFeedForward(nn.Module):
 
    def __init__(self, num_dim: int, dropout: float) -> None:
        super(PositionWiseFeedForward, self).__init__()

        self._W1 = nn.Linear(num_dim, 4 * num_dim)
        self._W2 = nn.Linear(4 * num_dim, num_dim)
        self._dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        # x -> (batch_size, context_length, n_dim)
        return self._dropout(self._W2(F.gelu(self._W1(x))))


class Transformer(nn.Module):
 
    def __init__(self, n_dim: int, n_heads: int, dropout: float) -> None:
        super(Transformer, self).__init__()

        self._multi_head_self_attention = MultiHeadSelfAttention(n_dim, n_heads, dropout)
        self._ffn = PositionWiseFeedForward(n_dim, dropout)
        self._norm1 = nn.LayerNorm(n_dim)
        self._norm2 = nn.LayerNorm(n_dim)
 

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = self._norm1(x + self._multi_head_self_attention(x, pad_mask))
        x = self._norm2(x + self._ffn(x))
 
        return x


class BERT4Rec(nn.Module): 
 
    def __init__(
        self,
        num_movies: int,
        context_length: int = 200,
        n_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super(BERT4Rec, self).__init__()
 
        self._num_movies = num_movies
        self._pad_token = 0
        self._mask_token = self._num_movies + 1
        
        vocab_size = self._num_movies + 2
 
        self.context_length = context_length
        self._n_dim = n_dim
         
        self._movie_embedding = nn.Embedding(vocab_size, n_dim, padding_idx=self._pad_token)
        self._positional_embedding = nn.Embedding(context_length, n_dim)

        self._embedding_dropout = nn.Dropout(dropout)
        self._embedding_normalization = nn.LayerNorm(n_dim)
 
        self._transformer_layers = nn.ModuleList([Transformer(n_dim, n_heads, dropout) for _ in range(n_layers)])
 
        self._output_projection = nn.Linear(n_dim, n_dim)
        self._output_bias = nn.Parameter(torch.zeros(vocab_size))

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias) 

 
    def _make_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens == self._pad_token).unsqueeze(1).unsqueeze(2)
 
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, context_length = tokens.shape

        positions = torch.arange(context_length, device=tokens.device).unsqueeze(0)  # (1, context_length)
        embeddings = self._movie_embedding(tokens) + self._positional_embedding(positions)
        embeddings = self._embedding_normalization(self._embedding_dropout(embeddings))  # (batch_size, context_length, n_dim)
 
        padding_mask = self._make_pad_mask(tokens)  # (batch_size, 1, 1, context_length)
 
        for transformer_layer in self._transformer_layers:
            embeddings = transformer_layer(embeddings, padding_mask) # (batch_size, context_length, n_dim)

        embeddings = self._output_projection(embeddings)  # (batch_size, context_length, n_dim) 
        logits = embeddings @ self._movie_embedding.weight.T + self._output_bias  # (batch_size, context_length, vocab_size)
 
        return logits
    
    
    @torch.no_grad()
    def recommend(
        self, 
        tokens: torch.Tensor, # (batch_size, context_length) sequence ending with [MASK]
        k: int = 10
    ) -> torch.Tensor:
        logits = self.forward(tokens)  # (batch_size, context_length, vocab_size)
        last_logits = logits[:, -1, :]  # (batch_size, vocab_size)
 
        last_logits[:, self._pad_token] = float("-inf")
        last_logits[:, self._mask_token] = float("-inf")
 
        return torch.topk(last_logits, k, dim=-1).indices  # (batch_size, k)
 