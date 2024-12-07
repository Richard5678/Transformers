import unittest
from transformers.blocks import DecoderBlock 
import torch


class TestBlocks(unittest.TestCase):
    
    def test_shape(self):
        embed_dim = 10
        bat_size = 50
    
        decoder = DecoderBlock(embed_dim=embed_dim, num_heads=5, hidden_dim=3)
        input = torch.rand((bat_size, embed_dim))
        output = decoder(input)

        self.assertEqual(input.shape, output.shape)