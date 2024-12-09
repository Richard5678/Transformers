import unittest
from transformers.blocks import MultiHeadAttentionBlock, FeedForwardNetworkBlock
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestMultiHeadAttention(unittest.TestCase):
    def test_shape(self):
        embed_dim = 10
        bat_size = 50
    
        multi_head_attn = MultiHeadAttentionBlock(embed_dim=embed_dim, num_heads=5).to(device)
        input = torch.rand((bat_size, embed_dim)).to(device)
        output = multi_head_attn(input, input, input)

        self.assertEqual(input.shape, output.shape)

        
class TestFeedForwardNetwork(unittest.TestCase):
    def test_shape(self):
        embed_dim = 10
        bat_size = 50
    
        feed_forward_network = FeedForwardNetworkBlock(embed_dim=embed_dim, hidden_dim=3).to(device)
        input = torch.rand((bat_size, embed_dim)).to(device)
        output = feed_forward_network(input)

        self.assertEqual(input.shape, output.shape)