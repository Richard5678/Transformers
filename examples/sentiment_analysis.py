from transformers import AutoTokenizer
from mytransformers.models import TransformerEncoderOnly
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_data(dataset, tokenizer):
    X = tokenizer(dataset["text"], padding=True, truncation=True, return_tensors="pt")
    Y = torch.tensor(dataset["label"])

    return X, Y


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return self.encodings["input_ids"][idx], self.labels[idx]

    def __len__(self):
        return len(self.encodings["input_ids"])


def train_model(model, train_loader, epoch=1):
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for i in range(epoch):
        for input_ids, labels in tqdm(train_loader):
            optimizer.zero_grad()

            # forward pass
            input_ids, labels = input_ids.to(device), labels.to(device)
            output = model(input_ids)  # (batch_size, seq_len, num_labels)

            # loss calculation
            logits = output[:, 0, :]  # (batch_size, num_labels)
            loss = criterion(logits, labels)

            # backward pass
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader):
    model.eval()

    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        average_accuracy = average_loss = 0
        all_predictions = []
        all_labels = []

        for input_ids, labels in tqdm(test_loader):
            # forward pass
            input_ids, labels = input_ids.to(device), labels.to(device)
            output = model(input_ids)  # (batch_size, seq_length, num_labels)

            # loss calculation
            logits = output[:, 0, :]  # (batch_size, num_labels)
            loss = criterion(logits, labels)
            average_loss += loss.item()

            predictions = logits.argmax(dim=1)
            average_accuracy += (predictions == labels).float().mean()

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        average_loss /= len(test_loader)
        average_accuracy /= len(test_loader)

        print(f"average loss is: {average_loss}")
        print(f"average accuracy is: {average_accuracy}")

        matrix = confusion_matrix(
            np.concatenate(all_labels), np.concatenate(all_predictions)
        )

        print(f"confusion matrix:\n {matrix}")


def main():
    # load dataset
    dataset = load_dataset("imdb")

    # tokenization
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_encodings, train_labels = tokenize_data(dataset["train"], tokenizer)
    test_encodings, test_labels = tokenize_data(dataset["test"], tokenizer)

    # dataset construction
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # model creation
    model = TransformerEncoderOnly(
        embed_dim=768,
        num_heads=16,
        vocab_size=tokenizer.vocab_size,
        num_labels=2,
        max_seq_length=512,
        num_layers=3,
        hidden_dim=768,
    ).to(device)

    # model training
    train_model(model, train_loader)

    # model evaluation
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
