import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence order information"""
    
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # محاسبه positional encoding با استفاده از سینوس و کوسینوس
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)  # (maxlen, 1, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        # token_embedding: (seq_len, batch_size, emb_size)
        return self.dropout(token_embedding + 
                           self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """Convert token indices to dense embeddings"""
    
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (seq_len, batch_size)
        # output: (seq_len, batch_size, emb_size)
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
