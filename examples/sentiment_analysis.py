from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from mytransformers.models import TransformerEncoderOnly
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    dataset = load_dataset("imdb")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    return train_dataset, test_dataset


def tokenize_data(dataset, tokenizer):
    X = tokenizer(dataset["text"], padding=True, truncation=True, return_tensors="pt")
    Y = torch.tensor(dataset["label"])
    return X, Y


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_model(model, train_loader, epochs=10):
    optimizer = Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for i in range(epochs):
        for X in tqdm(train_loader):
            input_ids = X['input_ids'].to(device)
            labels = X['labels'].to(device)

            optimizer.zero_grad()
            with autocast():
                output = model(input_ids)
                logits = output[:, 0, :]
                loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {i+1} loss: {loss.item()}")


def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)

            # accuracy
            accuracy = (output.argmax(dim=1) == Y).float().mean()
            print(f"Test accuracy: {accuracy.item()}")

            # confusion matrix
            conf_matrix = confusion_matrix(
                Y.cpu().numpy(), output.argmax(dim=1).cpu().numpy()
            )
            print(f"Confusion matrix: {conf_matrix}")

            # loss calculation
            loss = criterion(output, Y)
            print(f"Test loss: {loss.item()}")


def main():
    # load dataset
    dataset = load_dataset("imdb")

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # tokenize datasets
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    train_encodings, train_labels = tokenize_data(train_dataset, tokenizer)
    test_encodings, test_labels = tokenize_data(test_dataset, tokenizer)

    # create custom datasets
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # create model
    model = TransformerEncoderOnly(
        embed_dim=768,
        num_heads=12,
        vocab_size=tokenizer.vocab_size,
        num_labels=2,
        max_seq_length=512,
        num_layers=12,
        hidden_dim=768,
    ).to(device)

    # train model
    train_model(model, train_loader)

    # evaluate model
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
