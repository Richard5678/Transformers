"""
This is a simple machine translation example (English to Chinese) using a TransformerEncoderDecoder model.

# run with:
# torchrun --nproc_per_node=8 examples/machine_translation_distributed.py
"""

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import math

from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        vocab_size_en,
        vocab_size_zh,
        embed_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
    ):
        super(TransformerEncoderDecoder, self).__init__()
        self.embedding_en = nn.Embedding(vocab_size_en, embed_dim)
        self.embedding_zh = nn.Embedding(vocab_size_zh, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size_zh)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding_en(src) * math.sqrt(self.transformer.d_model)
        tgt_emb = self.embedding_zh(tgt) * math.sqrt(self.transformer.d_model)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.transformer.decoder(
            tgt_emb, memory, tgt_key_padding_mask=tgt_mask
        )
        return self.fc_out(output)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    train_dataset = dataset["train"].select(range(10))
    test_dataset = dataset["test"].select(range(10))
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings_en, encodings_zh):
        self.encodings_en = encodings_en
        self.encodings_zh = encodings_zh

    def __getitem__(self, idx):
        # Convert the Encoding objects to tensors
        input_ids_en = self.encodings_en["input_ids"][idx]
        input_ids_zh = self.encodings_zh["input_ids"][idx]
        return input_ids_en, input_ids_zh

    def __len__(self):
        return len(self.encodings_en["input_ids"])


def setup_distributed():
    import os

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)

    print(f"Initialized process {rank} out of {world_size} on device {local_rank}")


def cleanup_distributed():
    dist.destroy_process_group()


def train_model(model, train_loader, epochs=10):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    # only rank 0 will print the progress bar
    if dist.get_rank() == 0:
        epoch_iterator = tqdm(range(epochs))
    else:
        epoch_iterator = range(epochs)

    epoch_losses = []  # List to store loss of each epoch

    for epoch in epoch_iterator:
        train_loader.sampler.set_epoch(epoch)

        print(f"Starting epoch {epoch} on rank {dist.get_rank()}")
        epoch_loss = 0  # Initialize epoch loss
        if dist.get_rank() == 0:
            batch_iterator = tqdm(train_loader)
        else:
            batch_iterator = train_loader

        for input_ids_en, input_ids_zh in batch_iterator:
            optimizer.zero_grad()
            input_ids_en = input_ids_en.to(device)
            input_ids_zh = input_ids_zh.to(device)

            # Ensure masks have the correct shape (batch_size, seq_len)
            src_mask = (input_ids_en != 0).to(device)
            tgt_mask = (input_ids_zh != 0).to(device)

            # Transpose the masks to match the expected shape
            src_mask = src_mask.transpose(0, 1)
            tgt_mask = tgt_mask.transpose(0, 1)

            # forward pass: (batch_size, seq_len), (batch_size, seq_len) -> (batch_size, seq_len, vocab_size)
            outputs = model(input_ids_en, input_ids_zh, src_mask, tgt_mask)

            # compute loss
            target_ids = torch.roll(input_ids_zh, shifts=-1, dims=1)
            target_ids[:, -1] = 0  # pad token
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))

            # backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # Accumulate loss

        # Store the average loss for the epoch
        epoch_losses.append(epoch_loss / len(train_loader))

        # Synchronize all processes at the end of each epoch
        dist.barrier()

    # Print average loss for each epoch
    if dist.get_rank() == 0:
        print("Average loss for each epoch:")
        print(epoch_losses)


def evaluate_model(model, test_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    average_loss = 0
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for input_ids_en, input_ids_zh in tqdm(test_loader, desc="Evaluating"):
            input_ids_en = input_ids_en.to(device)
            input_ids_zh = input_ids_zh.to(device)

            src_mask = (input_ids_en != 0).to(device)
            tgt_mask = (input_ids_zh != 0).to(device)

            src_mask = src_mask.transpose(0, 1)
            tgt_mask = tgt_mask.transpose(0, 1)
            outputs = model(input_ids_en, input_ids_zh, src_mask, tgt_mask)

            target_ids = torch.roll(input_ids_zh, shifts=-1, dims=1)
            target_ids[:, -1] = 0  # pad token
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            average_loss += loss.item()

            # compute accuracy
            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(input_ids_zh.cpu().numpy().flatten())

    average_loss /= len(test_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    average_accuracy = np.mean(all_preds == all_labels)
    print(f"Average loss: {average_loss:.4f}, Average accuracy: {average_accuracy:.4f}")


def generate_text(
    model, input_ids, max_length=512, start_token_id=None, eos_token_id=None
):
    model.eval()
    # Initialize tgt_ids with the start token, matching the shape of input_ids
    tgt_ids = torch.full(input_ids.shape, 0, dtype=torch.long).to(device)
    tgt_ids[:, 0] = start_token_id
    with torch.no_grad():
        for i in tqdm(range(max_length - 1)):  # Adjust range to max_length - 1
            outputs = model(input_ids, tgt_ids)
            next_token_logits = outputs[:, i, :]  # Use the current position i
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            tgt_ids[:, i + 1] = next_token_id  # Append the token to index i + 1
            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break

    print(tgt_ids)
    return tgt_ids


@record
def main():
    setup_distributed()

    # Load data
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    # dataset["train"] = dataset["train"].select(range(10))
    # dataset["test"] = dataset["test"].select(range(10))

    # Inspect the dataset structure
    # print(dataset["train"][0])  # Print the first item to understand the structure

    # Tokenize data
    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")

    max_seq_length = get_max_seq_length(percentile=95, plot=True)

    # Tokenize data using batch processing with progress bar
    train_encodings_en = tokenizer_en.batch_encode_plus(
        [
            item["translation"]["en"]
            for item in tqdm(dataset["train"], desc="Tokenizing English Train Data")
        ],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    train_encodings_zh = tokenizer_zh.batch_encode_plus(
        [
            item["translation"]["zh"]
            for item in tqdm(dataset["train"], desc="Tokenizing Chinese Train Data")
        ],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    test_encodings_en = tokenizer_en.batch_encode_plus(
        [
            item["translation"]["en"]
            for item in tqdm(dataset["test"], desc="Tokenizing English Test Data")
        ],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    test_encodings_zh = tokenizer_zh.batch_encode_plus(
        [
            item["translation"]["zh"]
            for item in tqdm(dataset["test"], desc="Tokenizing Chinese Test Data")
        ],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    print("tokenization done")

    train_dataset = CustomDataset(train_encodings_en, train_encodings_zh)
    test_dataset = CustomDataset(test_encodings_en, test_encodings_zh)

    # Modify DataLoader to use DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, num_workers=4, sampler=train_sampler
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4, sampler=test_sampler
    )

    # Create model
    # model = TransformerEncoderDecoder(
    #     vocab_size_en=tokenizer_en.vocab_size,
    #     vocab_size_zh=tokenizer_zh.vocab_size,
    #     embed_dim=768,
    #     num_heads=16,
    #     num_layers=3,
    #     dim_feedforward=768,
    #     dropout=0.1,
    # ).to(device)
    model = TransformerEncoderDecoder(
        vocab_size_en=tokenizer_en.vocab_size,
        vocab_size_zh=tokenizer_zh.vocab_size,
        embed_dim=768,
        num_heads=16,
        num_layers=3,
        dim_feedforward=768,
        dropout=0.1,
    ).to(device)

    # Wrap the model with DistributedDataParallel
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    # Train model
    train_model(model, train_loader)

    # Save model
    if dist.get_rank() == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # Save the model's state_dict instead of the model itself
        torch.save(model.module.state_dict(), f"model_{timestamp}.pth")

    # Evaluate model
    evaluate_model(model, test_loader)

    cleanup_distributed()


def get_max_seq_length(percentile=95, plot=False):
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")

    # find number of tokens in each sequence
    en_seq_lengths = [
        len(
            tokenizer_en.encode(
                item["translation"]["en"],
            )
        )
        for item in tqdm(train_dataset, desc="Tokenizing English Train Data")
    ]
    zh_seq_lengths = [
        len(
            tokenizer_zh.encode(
                item["translation"]["zh"],
            )
        )
        for item in tqdm(train_dataset, desc="Tokenizing Chinese Train Data")
    ]
    # print(f"Train dataset size: {len(train_dataset)}")
    # print(f"Test dataset size: {len(test_dataset)}")

    max_seq_length_95_en = int(
        np.percentile(en_seq_lengths, percentile)
    )  # 95% coverage
    max_seq_length_95_zh = int(
        np.percentile(zh_seq_lengths, percentile)
    )  # 95% coverage

    print(f"en seq lengths: {len(en_seq_lengths)}")
    print(f"zh seq lengths: {len(zh_seq_lengths)}")

    print(f"max seq length en: {max_seq_length_95_en}")
    print(f"max seq length zh: {max_seq_length_95_zh}")

    if plot:
        plt.hist(
            en_seq_lengths,
            bins=range(1, max(en_seq_lengths) + 1),
            alpha=0.5,
            label="English",
        )
        plt.hist(
            zh_seq_lengths,
            bins=range(1, max(zh_seq_lengths) + 1),
            alpha=0.5,
            label="Chinese",
        )
        plt.xlabel("Sequence Length")
        plt.ylabel("Frequency")
        plt.title("Sequence Length Distribution")
        plt.legend()
        plt.savefig("seq_length_distribution.png")
        plt.show()

    return max(max_seq_length_95_en, max_seq_length_95_zh)


if __name__ == "__main__":
    main()
