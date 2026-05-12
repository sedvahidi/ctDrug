"""
Ablation model with custom Cross-Attention mechanism
Results reported in Table 6
"""

import torch.nn as nn
from ..positional import PositionalEncoding, TokenEmbedding
from ..decoder import Decoder, DecoderLayer  # استفاده از Decoder سفارشی شما


class CrossAttentionAblationModel(nn.Module):
    """
    Ablation model with custom cross-attention.
    Uses DecoderLayer with CrossAttention conditioning.
    """
    
    def __init__(self, num_decoder_layers: int, emb_size: int,
                 src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward: int = 1024, dropout: float = 0.1, args=None):
        super().__init__()
        
        decoder_layer = DecoderLayer(
            d_model=emb_size,
            nhead=args.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conditioning_type="cross_attention",
            args=args
        )
        
        self.transformer_decoder = Decoder(decoder_layer, num_decoder_layers)
        
        tgt_vocab_size = src_vocab_size
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
    
    def forward(self, drugs, tgt_mask, tgt_padding_mask, target):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(drugs))
        s, b = drugs.size()
        memory = target.unsqueeze(0).repeat(s, 1, 1)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, tgt_padding_mask)
        return self.generator(outs)
