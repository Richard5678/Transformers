"""Model Architectures

Implementation of components of Transformer architecture:
    - Decoder block
    - Encoder block

"""

import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        
        self.multi_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.multi_attn(x, x, x, attn_mask=attn_mask)

        # add & norm
        x += self.dropout1(attn_output)
        x = self.norm1(x)
        
        ff_out = self.ff(x)
        
        # add & norm
        x += ff_out
        x = self.norm2(x)

        return x