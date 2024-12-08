"""Model Architectures

Implementation of components of Transformer architecture:
    - Multi-head attention
    - Feed-forward network

"""

import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.multi_head_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        attn_output, _ = self.multi_head_attn(query, key, value, attn_mask=attn_mask)

        # add & norm
        value += self.dropout1(attn_output)
        value = self.norm1(value)

        return value


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        ff_output = self.linear1(x)
        ff_output = self.relu(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)

        # add & norm
        x += ff_output
        x = self.norm(x)

        return x
