"""
This is a simple machine translation example (English to Chinese) using a TransformerEncoderDecoder model.
"""

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# from mytransformers.models import TransformerEncoderDecoder
from machine_translation_distributed import TransformerEncoderDecoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
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


def train_model(model, train_loader, epochs=1):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in tqdm(range(epochs)):
        # train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0  # Initialize epoch loss
        for input_ids_en, input_ids_zh in tqdm(train_loader, desc="Training"):
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



def generate_text(model, input_ids, max_length=512, eos_token_id=None):
    model.eval()
    # Initialize tgt_ids with the start token, matching the shape of input_ids
    tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")
    start_token_id = (
        tokenizer_zh.cls_token_id
    )  # Assuming the tokenizer has a CLS token as start
    tgt_ids = torch.full((input_ids.shape[0], max_length), 0, dtype=torch.long).to(
        device
    )
    tgt_ids[:, 0] = start_token_id
    src_mask = (input_ids != 0).to(device)
    tgt_mask = (tgt_ids != 0).to(device)
    src_mask = src_mask.transpose(0, 1)
    tgt_mask = tgt_mask.transpose(0, 1)
    print(f"src_mask shape: {src_mask.shape}")
    print(f"tgt_mask shape: {tgt_mask.shape}")
    print(f"input_ids shape: {input_ids.shape}")
    print(f"tgt_ids shape: {tgt_ids.shape}")
    with torch.no_grad():
        for i in tqdm(range(max_length - 1)):  # Adjust range to max_length - 1
            outputs = model(input_ids, tgt_ids, src_mask=src_mask, tgt_mask=tgt_mask)
            next_token_logits = outputs[:, i, :]  # Use the current position i
            next_token_logits[:, 0] = -float("inf")

            next_token_id = torch.argmax(next_token_logits, dim=-1)
            # tgt_ids[:, i + 1] = next_token_id  # Append the token to index i + 1
            tgt_ids[0, i + 1] = next_token_id
            print(f"next_token_id: {next_token_id}; eos_token_id: {eos_token_id}")
            if eos_token_id is not None and (next_token_id == eos_token_id).all():
                break
            tgt_mask[i + 1, 0] = 1

    print(tgt_ids)
    return tgt_ids


def main():
    # Load data
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)
    dataset["train"] = dataset["train"].select(range(10))
    dataset["test"] = dataset["test"].select(range(10))

    # Inspect the dataset structure
    # print(dataset["train"][0])  # Print the first item to understand the structure

    # Tokenize data
    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")

    # max_seq_length = get_max_seq_length(percentile=95, plot=True)
    max_seq_length = 71

    # Tokenize data using batch processing with progress bar
    train_encodings_en = tokenizer_en.batch_encode_plus(
        [
            item["translation"]["en"]
            for item in tqdm(
                dataset["train"], desc="Tokenizing English Train Data"
            )
        ],
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    train_encodings_zh = tokenizer_zh.batch_encode_plus(
        [
            item["translation"]["zh"]
            for item in tqdm(dataset["train"], desc="Tokenizing Chinese Train Data")
        ],
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    test_encodings_en = tokenizer_en.batch_encode_plus(
        [
            item["translation"]["en"]
            for item in tqdm(dataset["test"], desc="Tokenizing English Test Data")
        ],
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    test_encodings_zh = tokenizer_zh.batch_encode_plus(
        [
            item["translation"]["zh"]
            for item in tqdm(dataset["test"], desc="Tokenizing Chinese Test Data")
        ],
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    train_dataset = CustomDataset(train_encodings_en, train_encodings_zh)
    test_dataset = CustomDataset(test_encodings_en, test_encodings_zh)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Create model
    # model = TransformerEncoderDecoder(
    #     embed_dim=768,
    #     num_heads=16,
    #     vocab_size=tokenizer_en.vocab_size,
    #     num_labels=tokenizer_zh.vocab_size,
    #     max_seq_length=max_seq_length,
    #     num_layers=3,
    #     hidden_dim=768,
    # ).to(device)
    from machine_translation_distributed import TransformerEncoderDecoder
    model = TransformerEncoderDecoder(
        vocab_size_en=tokenizer_en.vocab_size,
        vocab_size_zh=tokenizer_zh.vocab_size,
        embed_dim=768,
        num_heads=16,
        num_layers=3,
        dim_feedforward=768,
        dropout=0.1,
    ).to(device)

    # Train model
    train_model(model, train_loader)

    # Save model
    model.save_model("model.pth")

    # Evaluate model
    evaluate_model(model, test_loader)


def generate_example(max_seq_length):
    tokenizer_zh = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = TransformerEncoderDecoder(
    #     embed_dim=768,
    #     num_heads=16,
    #     vocab_size=tokenizer_en.vocab_size,
    #     num_labels=tokenizer_zh.vocab_size,
    #     max_seq_length=max_seq_length,
    #     num_layers=3,
    #     hidden_dim=768,
    # ).to(device)
    # model.load_model("/home/richard/Transformers/model_2024-12-14_16:42:36.pth")
    # model.load_model("model_2024-12-14_19:08:39.pth")
    # model.load_model("model_2024-12-15_01:07:29.pth")
    # model.load_model("model_2024-12-15_01:25:58.pth")
    # model.load_model("model_2024-12-15_01:33:03.pth")
    # model.load_model("model_2024-12-15_02:27:16.pth")
    # model.load_model("model_2024-12-15_02:36:19.pth")
    # model.load_model("model_2024-12-15_02:42:26.pth")

    from machine_translation_distributed import TransformerEncoderDecoder

    model = TransformerEncoderDecoder(
        vocab_size_en=tokenizer_en.vocab_size,
        vocab_size_zh=tokenizer_zh.vocab_size,
        embed_dim=768,
        num_heads=16,
        num_layers=3,
        dim_feedforward=768,
        dropout=0.1,
    ).to(device)
    # model.load_model("model_2024-12-15_03:27:21.pth")  # lib transformer
    model.load_model("model_2024-12-15_07:02:19.pth") # library use - long training

    # dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    # test_encodings_en = tokenizer_en.batch_encode_plus(
    #     [item["translation"]["en"] for item in tqdm(dataset["test"], desc="Tokenizing English Test Data")],
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt"
    # )
    # test_encodings_zh = tokenizer_zh.batch_encode_plus(
    #     [item["translation"]["zh"] for item in tqdm(dataset["test"], desc="Tokenizing Chinese Test Data")],
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt"
    # )
    # test_dataset = CustomDataset(test_encodings_en, test_encodings_zh)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # prompt = "Hello, how are you?"
    # prompt = "beijing is a beautiful city"
    prompt = "Several years ago here at TED, Peter Skillman introduced a design challenge called the marshmallow..."
    input_ids = tokenizer_en.encode(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    ).to(device)

    print(f"input_ids shape: {input_ids.shape}")
    # eos_token_id = tokenizer_zh.eos_token_id  # Assuming the tokenizer has an EOS token
    # print(f"eos_token_id: {eos_token_id}")
    outputs = generate_text(
        model, input_ids, max_length=max_seq_length, eos_token_id=tokenizer_zh.sep_token_id
    )
    # print translated text
    print(tokenizer_zh.decode(outputs[0], skip_special_tokens=True))


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
    # main()

    generate_example(max_seq_length=71)
