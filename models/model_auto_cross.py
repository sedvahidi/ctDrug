import argparse
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
import dataset as md
import torch.utils.data as tud
import os.path
import glob
import math
from collections import Counter
from torch import Tensor
import io
import time

vocabulary = mv.tokens_struct()
PAD_IDX = vocabulary.pad
BOS_IDX = vocabulary.bos
EOS_IDX = vocabulary.eos
########################
# x1 -> y
#######################
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int , dropout:float = 0.1, args = None):
        super(Seq2SeqTransformer, self).__init__()
        
        decoder_layer = DecoderLayer(d_model=emb_size,
        nhead=args.nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout)
        
        self.transformer_decoder = Decoder(
        decoder_layer,
        num_layers=num_decoder_layers
        )

        tgt_vocab_size = src_vocab_size        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, drugs: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor, target: Tensor):

        tgt_emb = self.positional_encoding(self.tgt_tok_emb(drugs))

        # transpose to (S, B, E)
        tgt_emb = tgt_emb

        s, b = drugs.size()
        memory = target.unsqueeze(0).repeat(s, 1, 1)

        outs = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask
        )

        return self.generator(outs)
        
        
class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )

    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask
            )

        return output

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead,
                 dim_feedforward:int = 1024, dropout:float = 0.1, args = None):
        super(DecoderLayer, self).__init__()
        
        ###################start_change##################
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.2)
        
        # Cross-attention (target protein as memory)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.2)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_ff = nn.Dropout(dropout)
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu
        
        ###################end_change##################

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor=None , tgt_padding_mask: Tensor=None):
        
        ###################start_change##################
        #1. Masked self-attention
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_padding_mask
        )[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        #2. Cross-attention
        tgt2 = self.multihead_attn(
            tgt, memory, memory
        )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        #3. Feedforward
        tgt2 = self.linear1(tgt)
        tgt2 = self.activation(tgt2)
        tgt2 = self.dropout_ff(tgt2)
        tgt2 = self.linear2(tgt2)

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
        ###################end_change##################
    

######################################################################
# Text tokens are represented by using token embeddings. Positional
# encoding is added to the token embedding to introduce a notion of word
# order.
# 

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


######################################################################
# We create a ``subsequent word`` mask to stop a target word from
# attending to its subsequent words. We also create masks, for masking
# source and target padding tokens
# 

def generate_square_subsequent_mask(sz, DEVICE='cuda'):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(tgt, DEVICE='cuda'):
  tgt_seq_len = tgt.shape[0]
  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return tgt_mask, tgt_padding_mask


