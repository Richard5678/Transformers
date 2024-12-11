"""
Tests for the transformers.py file
"""

import torch
from mytransformers.models import (
    TransformerEncoderOnly,
    TransformerDecoderOnly,
    TransformerEncoderDecoder,
)
import unittest


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestTransformerEncoder(unittest.TestCase):
    def test_shape(self):
        embed_dim = 10
        bat_size = 50
        seq_len = 10
        vocab_size = 1000  # Example value
        max_seq_length = 20  # Example value
        hidden_dim = 512  # Example value

        transformer = TransformerEncoderOnly(
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=5,
            num_labels=vocab_size,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            hidden_dim=hidden_dim,
        ).to(device)

        input = torch.randint(0, vocab_size, (bat_size, seq_len), dtype=torch.long).to(
            device
        )
        output = transformer(input)

        expected_output_shape = (bat_size, seq_len, vocab_size)
        self.assertEqual(expected_output_shape, output.shape)


class TestTransformerDecoder(unittest.TestCase):
    def test_shape(self):
        embed_dim = 10
        bat_size = 50
        seq_len = 10
        vocab_size = 1000  # Example value
        max_seq_length = 20  # Example value
        hidden_dim = 512  # Example value

        transformer = TransformerDecoderOnly(
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=5,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            hidden_dim=hidden_dim,
        ).to(device)

        input = torch.randint(0, vocab_size, (bat_size, seq_len), dtype=torch.long).to(
            device
        )
        output = transformer(input)

        expected_output_shape = (bat_size, seq_len, vocab_size)
        self.assertEqual(expected_output_shape, output.shape)


class TestTransformer(unittest.TestCase):
    def test_shape(self):
        embed_dim = 10
        bat_size = 50
        seq_len = 10
        vocab_size = 1000  # Example value
        max_seq_length = 20  # Example value
        hidden_dim = 512  # Example value

        transformer = TransformerEncoderDecoder(
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=5,
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            hidden_dim=hidden_dim,
        ).to(device)

        input = torch.randint(0, vocab_size, (bat_size, seq_len), dtype=torch.long).to(
            device
        )
        output = transformer(input, input)

        expected_output_shape = (bat_size, seq_len, vocab_size)
        self.assertEqual(expected_output_shape, output.shape)


if __name__ == "__main__":
    unittest.main()
