"""
Simple letter to letter mapping example using decoder only transformer.
"""

import string
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from mytransformers.models import TransformerDecoderOnly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_letter_pairs():
    """Generate all adjacent pairs of letters in the format '{letter 1} - {letter 2}'.

    Args:
        None

    Returns:
        List[str]: A list of all adjacent pairs of letters in the format '{letter 1} - {letter 2}'.
    """
    letters = string.ascii_lowercase
    pairs = []
    for i in range(len(letters)):
        pair = f"{letters[i]} - {letters[(i + 1) % 26]}"
        pairs.append(pair)

    return pairs


def create_dataset():
    """Create a dataset of letter pairs with vocabulary and tensor data.

    Args:
        None

    Returns:
        Tuple[torch.Tensor, Dict[str, int], Dict[int, str], int]: The dataset, word to index mapping, index to word mapping, and vocabulary size.
    """
    # Get all letter pairs
    pairs = generate_letter_pairs()

    # Create vocabulary: special tokens + unique words from pairs
    vocab = ["<PAD>", "<BOS>", "<EOS>"] + list(set(" ".join(pairs).split()))
    vocab_size = len(vocab)

    # Create word to index and index to word mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # Convert pairs to tensor data
    tensor_data = []
    for pair in pairs:
        # Add BOS token at start
        sequence = ["<BOS>"] + pair.split() + ["<EOS>"]
        tensor_data.append(torch.tensor([word2idx[word] for word in sequence]))

    # Move tensor data to the correct device
    return torch.stack(tensor_data).to(device), word2idx, idx2word, vocab_size


def train_model(model: nn.Module, input_ids: torch.Tensor, epochs: int = 500):
    """Train the model.

    Args:
        model (nn.Module): The model to train.
        input_ids (torch.Tensor): The input ids (indices of the words in the vocabulary) to train on.
        alphabet (List[str]): The alphabet to use for the model.

    Returns:
        None
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss(ignore_index=alphabet.index('<PAD>'))
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    input_ids = input_ids.to(device)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_ids)

        target = torch.roll(input_ids, -1, dims=1)
        target[:, -1] = 0  # pad last position
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def generate_predictions(
    model: nn.Module, input_ids: torch.Tensor, idx2word: Dict[int, str]
):
    """Generate predictions from the model.

    Args:
        model (nn.Module): The model to use for prediction.
        input_ids (torch.Tensor): The input ids (indices of the words in the vocabulary) to predict on.
        idx2word (Dict[int, str]): The index to word mapping.

    Returns:
        List[str]: The predicted words.
    """
    model.eval()
    pred = model(input_ids[:, :-2].to(device))
    argmax_indices = pred[:, -1, :].argmax(dim=-1)

    return [idx2word[idx.item()] for idx in argmax_indices]


def main():
    # # Create dataset
    input_ids, word2idx, idx2word, vocab_size = create_dataset()

    # # Define the model
    model = TransformerDecoderOnly(
        vocab_size=vocab_size,
        num_layers=3,
        num_heads=4,
        hidden_dim=16,
        embed_dim=64,
        max_seq_length=10,
    ).to(device)

    # Train model
    train_model(model, input_ids)

    # Generate predictions
    predictions = generate_predictions(model, input_ids, idx2word)

    print(predictions)

    # TODO: Generate tokens autoregressively until EOS token is reached


if __name__ == "__main__":
    main()
