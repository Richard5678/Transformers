'''
    This file contains the variations of the transformer model:
        - Encoder-Only
        - Decoder-Only
        - Encoder-Decoder (Seq2Seq)
'''

from transformers.blocks import MultiHeadAttentionBlock, FeedForwardNetworkBlock
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_position_ids(input_ids):
    batch_size, seq_len = input_ids.size()
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

    return position_ids


class TransformerEncoderOnly(nn.Module):
    def __init__(self, embed_dim, num_heads, vocab_size, max_seq_length, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoderOnly, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)

        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttentionBlock(embed_dim, num_heads, dropout),
                FeedForwardNetworkBlock(embed_dim, hidden_dim, dropout)
            ) for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_ids):
        # Move input_ids to the same device as the model
        input_ids = input_ids.to(device)
        
        token_embed = self.token_embedding(input_ids)
        pos_embed = self.pos_embedding(get_position_ids(input_ids))
        x = token_embed + pos_embed

        # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.encoder_layers:
            x = layer[0](x, x, x)
            x = layer[1](x)

        # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)

        # linear & softmax
        x = self.linear(x)
        x = self.softmax(x)

        return x

        
class TransformerDecoderOnly(nn.Module):
    def __init__(self, embed_dim, num_heads, vocab_size, max_seq_length, num_layers, hidden_dim, dropout=0.1):
        super(TransformerDecoderOnly, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)

        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttentionBlock(embed_dim, num_heads, dropout),
                FeedForwardNetworkBlock(embed_dim, hidden_dim, dropout)
            ) for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids):
        # Move input_ids to the same device as the model
        input_ids = input_ids.to(device)
        
        token_embed = self.token_embedding(input_ids)
        pos_embed = self.pos_embedding(get_position_ids(input_ids))
        x = token_embed + pos_embed

        # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        
        # decoder layers
        for layer in self.decoder_layers:
            x = layer[0](x, x, x)
            x = layer[1](x)

        # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)

        # linear & softmax
        x = self.linear(x)
        x = self.softmax(x)
        
        return x

        
class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, vocab_size, max_seq_length, num_layers, hidden_dim, dropout=0.1):
        super(TransformerEncoderDecoder, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_dim)

        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttentionBlock(embed_dim, num_heads, dropout),
                FeedForwardNetworkBlock(embed_dim, hidden_dim, dropout)
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

        self.self_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttentionBlock(embed_dim, num_heads, dropout)
        self.ff = FeedForwardNetworkBlock(embed_dim, hidden_dim, dropout)
        
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_ids, tgt_ids):
        # Move input_ids to the same device as the model
        input_ids = input_ids.to(device)
        tgt_ids = tgt_ids.to(device)
        
        # encoder embedding
        token_embed = self.token_embedding(input_ids)
        pos_embed = self.pos_embedding(get_position_ids(input_ids))
        x = token_embed + pos_embed

        # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.encoder_layers:
            x = layer[0](x, x, x)
            x = layer[1](x)
            
        memory = x

        # decoder embedding
        token_embed = self.token_embedding(tgt_ids)
        pos_embed = self.pos_embedding(get_position_ids(tgt_ids))
        x = token_embed + pos_embed

        # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)

        # decoder layers
        for _ in range(self.num_layers):
            x = self.self_attention(x, x, x)
            x = self.cross_attention(x, memory, memory)
            x = self.ff(x)

        # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)

        x = self.linear(x)
        x = self.softmax(x)

        return x
