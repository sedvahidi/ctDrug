import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """Cross-attention conditioning (original method)"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    
    def forward(self, tgt, memory):
        # tgt: (S, B, E)
        # memory: (S, B, E) or (1, B, E)
        return self.cross_attn(tgt, memory, memory)[0]


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, d_model)
        self.beta = nn.Linear(cond_dim, d_model)

    def forward(self, x, cond):
        # x: (S, B, E)
        # cond: (B, E)
        gamma = self.gamma(cond).unsqueeze(0)  # (1, B, E)
        beta = self.beta(cond).unsqueeze(0)    # (1, B, E)
        return gamma * x + beta


class ConcatConditioning(nn.Module):
    """Concatenation-based conditioning"""
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.proj = nn.Linear(d_model + cond_dim, d_model)

    def forward(self, x, cond):
        # x: (S, B, E)
        # cond: (B, E)
        cond = cond.unsqueeze(0).expand(x.size(0), -1, -1)  # (S, B, E)
        x_cat = torch.cat([x, cond], dim=-1)  # (S, B, 2E)
        return self.proj(x_cat)  # (S, B, E)
