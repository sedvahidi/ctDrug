import copy
import torch.nn as nn
import torch.nn.functional as F
from .conditioning import CrossAttention, FiLM, ConcatConditioning


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, 
                 conditioning_type="cross_attention", args=None):
        super().__init__()
        
        # 1. Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 2. Conditioning mechanism (بر اساس پارامتر)
        self.conditioning_type = conditioning_type
        
        if conditioning_type == "cross_attention":
            self.conditioning = CrossAttention(d_model, nhead, dropout)
        elif conditioning_type == "film":
            self.conditioning = FiLM(d_model, d_model)
        elif conditioning_type == "concatenation":
            self.conditioning = ConcatConditioning(d_model, d_model)
        else:
            raise ValueError(f"Unknown conditioning_type: {conditioning_type}")
        
        # 3. Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 4. Normalization & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        # 1. Masked self-attention
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 2. Conditioning (روش متفاوت بر اساس type)
        if self.conditioning_type == "cross_attention":
            # cross-attention: memory به طور کامل استفاده می‌شود
            tgt2 = self.conditioning(tgt, memory)
        else:
            # FiLM یا Concatenation: فقط یک embedding در هر batch استفاده می‌شود
            cond = memory[0]  # (B, E) - اولین زمان‌استپ
            tgt2 = self.conditioning(tgt, cond)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 3. Feedforward
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, 
                          tgt_padding_mask=tgt_padding_mask)
        return output
