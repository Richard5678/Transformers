"""
This is a simple machine translation example (English to Chinese) using a TransformerEncoderDecoder model.

# run with:
# torchrun --nproc_per_node=4 examples/machine_translation_distributed.py
"""

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import math
import socket
import copy

from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from torch import nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

# from mytransformers.models import TransformerEncoderDecoder

import os
import warnings
import signal
import sys

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# encoder decoder transformer built on top of library implementation of transformer
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
        max_seq_length=104,
    ):
        super(TransformerEncoderDecoder, self).__init__()
        self.embedding_en = nn.Embedding(vocab_size_en, embed_dim)
        self.embedding_zh = nn.Embedding(vocab_size_zh, embed_dim)
        self.pos_embedding_en = nn.Embedding(max_seq_length, embed_dim)
        self.pos_embedding_zh = nn.Embedding(max_seq_length, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size_zh)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_pos = (
            torch.arange(0, src.size(1), device=src.device)
            .unsqueeze(0)
            .repeat(src.size(0), 1)
        )
        tgt_pos = (
            torch.arange(0, tgt.size(1), device=tgt.device)
            .unsqueeze(0)
            .repeat(tgt.size(0), 1)
        )

        src_emb = self.embedding_en(src) * math.sqrt(
            self.transformer.d_model
        ) + self.pos_embedding_en(src_pos)
        tgt_emb = self.embedding_zh(tgt) * math.sqrt(
            self.transformer.d_model
        ) + self.pos_embedding_zh(tgt_pos)

        # src_emb = self.embedding_en(src) * math.sqrt(self.transformer.d_model)
        # tgt_emb = self.embedding_zh(tgt) * math.sqrt(self.transformer.d_model)

        # Ensure masks are of the same type
        src_key_padding_mask = (
            src_key_padding_mask.to(torch.bool)
            if src_key_padding_mask is not None
            else None
        )
        tgt_key_padding_mask = (
            tgt_key_padding_mask.to(torch.bool)
            if tgt_key_padding_mask is not None
            else None
        )

        subsequent_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool), diagonal=1
        ).to(device)

        output = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=subsequent_mask,
        )
        return self.fc_out(output)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


def load_data():
    """Loads the dataset and returns train and test datasets
    Returns:
        train_dataset: torch.utils.data.Dataset
        test_dataset: torch.utils.data.Dataset
    """
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


# Torch dataset that return pairs of input ids for english and chinese
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


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed():
    # Only set MASTER_PORT if not already set
    if "MASTER_PORT" not in os.environ:
        free_port = find_free_port()
        os.environ["MASTER_PORT"] = str(free_port)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)

    print(
        f"Initialized process {rank} out of {world_size} on device {local_rank} with MASTER_PORT={os.environ['MASTER_PORT']}"
    )


def cleanup_distributed():
    dist.destroy_process_group()


def get_learning_rate(step, d_model, warmup_steps):
    """Compute learning rate using the formula from the Transformer paper."""
    arg1 = step**-0.5
    arg2 = step * (warmup_steps**-1.5)
    return (d_model**-0.5) * min(arg1, arg2)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    d_model=512,
    warmup_steps=4000,
    save_every=5,
    start_epoch=0,
    checkpoint_dir="checkpoints",
    patience=5,
):
    """
    Model training loop with early stopping based on validation loss.
    """
    optimizer = AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    epoch_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0  # Track the global step number
    best_model_state = copy.deepcopy(model.state_dict())  # To store the best model

    # Train loop
    # for epoch in tqdm(range(epochs), desc="Training epochs"):
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)

        print(f"Starting epoch {epoch} on rank {dist.get_rank()}")
        epoch_loss = 0
        if dist.get_rank() == 0:
            batch_iterator = tqdm(train_loader)  # Progress bar for training
        else:
            batch_iterator = train_loader

        for input_ids_en, input_ids_zh in batch_iterator:
            optimizer.zero_grad()
            input_ids_en = input_ids_en.to(device)
            input_ids_zh = input_ids_zh.to(device)

            src_mask = (input_ids_en == 0).to(device)
            tgt_mask = (input_ids_zh == 0).to(device)

            outputs = model(input_ids_en, input_ids_zh, src_mask, tgt_mask)

            target_ids = input_ids_zh[:, 1:].reshape(-1)
            output_logits = outputs[:, :-1, :].reshape(-1, outputs.size(-1))

            loss = criterion(output_logits, target_ids)
            loss.backward()

            # Update learning rate
            lr = get_learning_rate(global_step + 1, d_model, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1  # Increment the global step

        epoch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} training loss: {epoch_loss:.4f}")

        # Validate after each epoch
        val_loss = evaluate_loss(model, val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1} validation loss: {val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print("Validation loss improved. Saving best model state.")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement in validation loss for {epochs_without_improvement} epoch(s)."
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"best_model_{epoch + 1 + start_epoch}.pth"
            )
            torch.save(best_model_state, checkpoint_path)
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

        # Save checkpoint every 'save_every' epochs
        if (epoch + 1) % save_every == 0 and dist.get_rank() == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1 + start_epoch}.pth"
            )
            torch.save(best_model_state, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

        dist.barrier()

    # Restore the best model state
    model.load_state_dict(best_model_state)
    print("Restored the best model state based on validation loss.")

    dist.barrier()

    if dist.get_rank() == 0:
        print("Training complete.")
        print("Training losses:", epoch_losses)
        print("Validation losses:", val_losses)


def evaluate_model(model, test_loader):
    """Evaluate the model on the test set
    Args:
        model: The model to evaluate
        test_loader: The testing data loader
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    average_loss = 0
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for input_ids_en, input_ids_zh in tqdm(test_loader, desc="Evaluating"):
            input_ids_en = input_ids_en.to(device)
            input_ids_zh = input_ids_zh.to(device)

            src_key_padding_mask = (input_ids_en == 0).to(device)
            tgt_key_padding_mask = (input_ids_zh == 0).to(device)

            outputs = model(
                input_ids_en, input_ids_zh, src_key_padding_mask, tgt_key_padding_mask
            )

            target_ids = input_ids_zh[:, 1:].contiguous()
            output_logits = outputs[:, :-1, :].contiguous()
            loss = criterion(
                output_logits.view(-1, output_logits.size(-1)), target_ids.view(-1)
            )
            average_loss += loss.item()

            # compute accuracy
            preds = torch.argmax(output_logits, dim=-1)
            non_pad_indices = target_ids != 0
            all_preds.extend(preds[non_pad_indices].cpu().numpy().flatten())
            all_labels.extend(target_ids[non_pad_indices].cpu().numpy().flatten())

    average_loss /= len(test_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    average_accuracy = np.mean(all_preds == all_labels)
    print(f"Average loss: {average_loss:.4f}, Average accuracy: {average_accuracy:.4f}")


def generate_text(
    model, input_ids, max_length=512, start_token_id=None, eos_token_id=None
):
    """Generate text using the model
    Args:
        model: The model to use for generation
        input_ids: The input IDs
        max_length: The maximum length of the generated text
        start_token_id: The start token ID
        eos_token_id: The end of sequence token ID
    Returns:
        tgt_ids: The generated text
    """
    model.eval()
    # Initialize tgt_ids with the start token, matching the shape of input_ids
    tgt_ids = torch.full(input_ids.shape, 0, dtype=torch.long).to(device)
    tgt_ids[:, 0] = start_token_id
    src_key_padding_mask = (input_ids == 0).to(device)
    tgt_key_padding_mask = (tgt_ids == 0).to(device)
    with torch.no_grad():
        for i in tqdm(range(max_length - 1)):  # Adjust range to max_length - 1
            outputs = model(
                input_ids, tgt_ids, src_key_padding_mask, tgt_key_padding_mask
            )
            next_token_logits = outputs[:, i, :]  # Use the current position i
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            tgt_ids[:, i + 1] = next_token_id  # Append the token to index i + 1
            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break

            tgt_key_padding_mask[:, i + 1] = False

    print(tgt_ids)
    return tgt_ids


def signal_handler(sig, frame):
    print("Terminating...")
    dist.destroy_process_group()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_encodings(tokenizer, dataset, lang="en", max_seq_length=104):
    """Tokenize the dataset
    Args:
        tokenizer: The tokenizer to use
        dataset: The dataset to tokenize
        lang: The language to tokenize
    Returns:
        encodings: The tokenized dataset
    """
    encodings = tokenizer.batch_encode_plus(
        [item["translation"][lang] for item in dataset],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    return encodings


def preprocess_data():
    """Preprocess the dataset
    Returns:
        train_loader: torch.utils.data.DataLoader
        test_loader: torch.utils.data.DataLoader
    """
    train_dataset, test_dataset = load_data()

    # dataset["train"] = dataset["train"].select(range(10))
    # dataset["test"] = dataset["test"].select(range(10))

    # Tokenize data
    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")

    # max_seq_length = get_max_seq_length(percentile=100, plot=True)
    # print(f"max_seq_length: {max_seq_length}")
    # return
    max_seq_length = 104  # 99 percentile

    # Tokenize data using batch processing with progress bar
    train_encodings_en = get_encodings(
        tokenizer_en, train_dataset, lang="en", max_seq_length=max_seq_length
    )
    train_encodings_zh = get_encodings(
        tokenizer_zh, train_dataset, lang="zh", max_seq_length=max_seq_length
    )
    test_encodings_en = get_encodings(
        tokenizer_en, test_dataset, lang="en", max_seq_length=max_seq_length
    )
    test_encodings_zh = get_encodings(
        tokenizer_zh, test_dataset, lang="zh", max_seq_length=max_seq_length
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

    return train_loader, test_loader, tokenizer_en, tokenizer_zh, max_seq_length


def get_max_seq_length(percentile=95, plot=False):
    """Get the maximum sequence length for the dataset
    Find the maximum of the 'percentile' percentile of the english and chinese sequence lengths
    """
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


def evaluate_loss(model, val_loader):
    """
    Evaluate the model on the validation set and return the average loss.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for input_ids_en, input_ids_zh in tqdm(
            val_loader, desc="Evaluating Validation Loss"
        ):
            input_ids_en = input_ids_en.to(device)
            input_ids_zh = input_ids_zh.to(device)

            src_key_padding_mask = (input_ids_en == 0).to(device)
            tgt_key_padding_mask = (input_ids_zh == 0).to(device)

            outputs = model(
                input_ids_en, input_ids_zh, src_key_padding_mask, tgt_key_padding_mask
            )

            target_ids = input_ids_zh[:, 1:].contiguous()
            output_logits = outputs[:, :-1, :].contiguous()
            loss = criterion(
                output_logits.view(-1, output_logits.size(-1)), target_ids.view(-1)
            )
            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)
    model.train()
    return average_loss


@record
def main():
    # Setup distributed training
    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Only the process with rank 0 creates the checkpoint directory
    if rank == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_dir = f"checkpoints_{timestamp}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory created at: {checkpoint_dir}")
    else:
        checkpoint_dir = None  # Other ranks do not create the directory

    # Ensure all processes wait until the checkpoint directory is created
    dist.barrier()

    # Load data
    cache_dir = "~/data"
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    train_loader, test_loader, tokenizer_en, tokenizer_zh, max_seq_length = (
        preprocess_data()
    )

    # Split the test_loader into validation and test sets
    # For example purposes, we'll use a simple split
    val_size = int(0.5 * len(test_loader.dataset))
    test_size = len(test_loader.dataset) - val_size
    val_loader, test_loader = torch.utils.data.random_split(
        test_loader.dataset, [val_size, test_size]
    )

    # Create new DataLoader instances
    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
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
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_length=max_seq_length,
    ).to(device)
    # model.load_model('model_2024-12-19_07:41:27.pth') # 50 epochs

    # model = TransformerEncoderDecoder(
    #     embed_dim=512,
    #     num_heads=8,
    #     vocab_size=tokenizer_en.vocab_size,
    #     num_labels=tokenizer_zh.vocab_size,
    #     max_seq_length=max_seq_length,
    #     num_layers=6,
    #     hidden_dim=512,
    # ).to(device)

    # Wrap the model with DistributedDataParallel
    # model = DDP(model, device_ids=[torch.cuda.current_device()])

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,  # Pass validation loader
        start_epoch=0,
        checkpoint_dir=checkpoint_dir,
        patience=5,  # Set patience parameter
    )

    # Save the best model
    if dist.get_rank() == 0:
        model_checkpoint_path = os.path.join(
            checkpoint_dir, f"best_model_{timestamp}.pth"
        )
        torch.save(model.module.state_dict(), model_checkpoint_path)
        print(f"Best model saved at: {model_checkpoint_path}")

    # Evaluate model
    evaluate_model(model, test_loader)

    cleanup_distributed()


if __name__ == "__main__":
    main()
