import torch.nn as nn
from .decoder import Decoder, DecoderLayer
from .positional import PositionalEncoding, TokenEmbedding


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward: int, dropout: float = 0.1, args=None):
        super().__init__()
        
        # دریافت conditioning_type از args
        conditioning_type = getattr(args, 'conditioning_type', 'cross_attention')
        
        decoder_layer = DecoderLayer(
            d_model=emb_size,
            nhead=args.nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            conditioning_type=conditioning_type,
            args=args
        )
        
        self.transformer_decoder = Decoder(decoder_layer, num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
    
    def forward(self, drugs, tgt_mask, tgt_padding_mask, target):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(drugs))
        s, b = drugs.size()
        memory = target.unsqueeze(0).repeat(s, 1, 1)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                        tgt_padding_mask=tgt_padding_mask)
        return self.generator(outs)
