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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import jieba

from mytransformers.models import TransformerEncoderDecoder

# from machine_translation_distributed import TransformerEncoderDecoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class TransformerEncoderDecoder(nn.Module):
#     def __init__(
#         self,
#         vocab_size_en,
#         vocab_size_zh,
#         embed_dim,
#         num_heads,
#         num_layers,
#         dim_feedforward,
#         dropout,
#     ):
#         super(TransformerEncoderDecoder, self).__init__()
#         self.embedding_en = nn.Embedding(vocab_size_en, embed_dim)
#         self.embedding_zh = nn.Embedding(vocab_size_zh, embed_dim)
#         self.transformer = nn.Transformer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation="relu",
#             batch_first=True,
#         )
#         self.fc_out = nn.Linear(embed_dim, vocab_size_zh)

#     def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
#         src_emb = self.embedding_en(src) * math.sqrt(self.transformer.d_model)
#         tgt_emb = self.embedding_zh(tgt) * math.sqrt(self.transformer.d_model)

#         # Ensure masks are of the same type
#         src_key_padding_mask = (
#             src_key_padding_mask.to(torch.bool)
#             if src_key_padding_mask is not None
#             else None
#         )
#         tgt_key_padding_mask = (
#             tgt_key_padding_mask.to(torch.bool)
#             if tgt_key_padding_mask is not None
#             else None
#         )

#         subsequent_mask = torch.triu(
#             torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool), diagonal=1
#         ).to(device)

#         output = self.transformer(
#             src_emb,
#             tgt_emb,
#             src_key_padding_mask=src_key_padding_mask,
#             tgt_key_padding_mask=tgt_key_padding_mask,
#             tgt_mask=subsequent_mask,
#         )
#         return self.fc_out(output)

#     def load_model(self, path):
#         self.load_state_dict(torch.load(path))


def load_data():
    """Load the dataset
    Returns:
        train_dataset: The training dataset
        test_dataset: The testing dataset
    """
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset for the machine translation task
    Args:
        encodings_en: The English encodings
        encodings_zh: The Chinese encodings
    Returns:
        input_ids_en: The English input IDs
        input_ids_zh: The Chinese input IDs
    """

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
    """Train the model
    Args:
        model: The model to train
        train_loader: The training data loader
        epochs: The number of epochs to train for
    """
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
            src_mask = (input_ids_en == 0).to(device)
            tgt_mask = (input_ids_zh == 0).to(device)

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
    """Evaluate the model
    Args:
        model: The model to evaluate
        test_loader: The testing data loader
    Returns:
        average_loss: The average loss
        average_accuracy: The average accuracy
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

            src_mask = (input_ids_en == 0).to(device)
            tgt_mask = (input_ids_zh == 0).to(device)

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
) -> torch.Tensor:
    """Generate text using the given model
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

    # print(tgt_ids)
    return tgt_ids


def get_bleu_score(
    target_seq: torch.Tensor, pred_seq: torch.Tensor, tokenizer: AutoTokenizer
) -> float:
    """Get the BLEU score for the given target and predicted sequences
    Args:
        target_seq: The target sequence
        pred_seq: The predicted sequence
        tokenizer: The tokenizer to use
    Returns:
        bleu_score: The BLEU score
    """

    # Convert token IDs back to text
    if isinstance(target_seq, torch.Tensor):
        target_seq = target_seq.cpu().numpy()
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.cpu().numpy()

    # Remove padding tokens (0s)
    target_seq = target_seq[target_seq != 0]
    pred_seq = pred_seq[pred_seq != 0]

    # Convert to strings and segment Chinese text
    target_str = tokenizer.decode(target_seq, skip_special_tokens=True)
    pred_str = tokenizer.decode(pred_seq, skip_special_tokens=True)

    # Segment into words/characters
    target_tokens = list(jieba.cut(target_str))
    pred_tokens = list(jieba.cut(pred_str))

    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)  # Equal weights for 1-4 grams

    try:
        bleu_score = sentence_bleu(
            [target_tokens], pred_tokens, weights=weights, smoothing_function=smoothing
        )
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        bleu_score = 0.0

    return bleu_score


def evaluate_model_bleu(model, test_loader, tokenizer_zh):
    """Evaluate the model using BLEU score
    Args:
        model: The model to evaluate
        test_loader: The testing data loader
        tokenizer_zh: The Chinese tokenizer
    Returns:
        average_bleu_score: The average BLEU score
        std_bleu_score: The standard deviation of the BLEU scores
    """
    bleu_scores = []
    for input_ids_en, input_ids_zh in tqdm(test_loader, desc="Evaluating"):
        input_ids_en = input_ids_en.to(device)
        input_ids_zh = input_ids_zh.to(device)

        target_ids = torch.roll(input_ids_zh, shifts=-1, dims=1).to(device)
        target_ids[:, -1] = 0  # pad token

        pred_seq = generate_text(
            model,
            input_ids_en,
            max_length=max_seq_length,
            start_token_id=tokenizer_zh.cls_token_id,
            eos_token_id=tokenizer_zh.sep_token_id,
        )
        # print(f"Target ids: {target_ids}")
        # print(f"Pred ids: {pred_seq}")
        bleu_score = get_bleu_score(target_ids, pred_seq, tokenizer_zh)
        bleu_scores.append(bleu_score)

    average_bleu_score = np.mean(bleu_scores)
    std_bleu_score = np.std(bleu_scores)
    print(f"Average BLEU score: {average_bleu_score:.4f} ± {std_bleu_score:.4f}")


def main():
    """Main function to train and evaluate the model"""
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
    # max_seq_length = 71
    max_seq_length = 104

    # Tokenize data using batch processing with progress bar
    train_encodings_en = tokenizer_en.batch_encode_plus(
        [
            item["translation"]["en"]
            for item in tqdm(dataset["train"], desc="Tokenizing English Train Data")
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

    # model = TransformerEncoderDecoder(
    #     vocab_size_en=tokenizer_en.vocab_size,
    #     vocab_size_zh=tokenizer_zh.vocab_size,
    #     embed_dim=768,
    #     num_heads=16,
    #     num_layers=3,
    #     dim_feedforward=768,
    #     dropout=0.1,
    # ).to(device)
    # model.load_model("model_2024-12-15_03:27:21.pth")  # lib transformer
    # model.load_model("model_2024-12-15_07:02:19.pth") # library use - long training

    model = TransformerEncoderDecoder(
        vocab_size_en=tokenizer_en.vocab_size,
        vocab_size_zh=tokenizer_zh.vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)
    # model.load_model("model_2024-12-18_03:40:35.pth")
    # model.load_model("model_2024-12-18_04:07:10.pth") # 2 epoch
    # model.load_model("model_2024-12-18_11:57:17.pth")  # 50 epochs 95% truncation
    model.load_model("model_2024-12-19_07:41:27.pth")  # 50 epochs 99% truncation
    # model.load_model("checkpoints_2025-01-05_16-23-13/checkpoint_epoch_85.pth")

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

    # lib transformers with positional embeddings

    # model = TransformerEncoderDecoder(
    #     vocab_size_en=tokenizer_en.vocab_size,
    #     vocab_size_zh=tokenizer_zh.vocab_size,
    #     embed_dim=512,
    #     num_heads=8,
    #     num_layers=6,
    #     dim_feedforward=512,
    #     dropout=0.1,
    #     max_seq_length=max_seq_length,
    # ).to(device)
    # model.load_model("checkpoints_2025-01-06_05-23-31/checkpoint_epoch_40.pth")

    # prompt = "hello, how are you?"
    # prompt = "beijing is a beautiful city"
    prompt = "Several years ago here at TED, Peter Skillman introduced a design challenge called the marshmallow..."
    target = "几年前，彼得·斯金曼在这里的TED上介绍了一个名为棉花糖挑战的设计挑战。"
    # prompt = "I am a student at the University of California, Berkeley"
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
        model,
        input_ids,
        max_length=max_seq_length,
        start_token_id=tokenizer_zh.cls_token_id,
        eos_token_id=tokenizer_zh.sep_token_id,
    )
    # print translated text
    print(tokenizer_zh.decode(outputs[0], skip_special_tokens=True))

    target_ids = tokenizer_zh.encode(
        target,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    ).to(device)
    print(f"BLEU score: {get_bleu_score(target_ids, outputs, tokenizer_zh)}")


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

    generate_example(max_seq_length=104)

    exit()

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

    # model = TransformerEncoderDecoder(
    #     vocab_size_en=tokenizer_en.vocab_size,
    #     vocab_size_zh=tokenizer_zh.vocab_size,
    #     embed_dim=768,
    #     num_heads=16,
    #     num_layers=3,
    #     dim_feedforward=768,
    #     dropout=0.1,
    # ).to(device)
    # model.load_model("model_2024-12-15_03:27:21.pth")  # lib transformer
    # model.load_model("model_2024-12-15_07:02:19.pth") # library use - long training

    model = TransformerEncoderDecoder(
        vocab_size_en=tokenizer_en.vocab_size,
        vocab_size_zh=tokenizer_zh.vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)
    # model.load_model("model_2024-12-18_03:40:35.pth")
    # model.load_model("model_2024-12-18_04:07:10.pth") # 2 epoch
    # model.load_model("model_2024-12-18_11:57:17.pth")  # 50 epochs 95% truncation
    model.load_model("model_2024-12-19_07:41:27.pth")

    max_seq_length = 104

    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")

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

    test_dataset = CustomDataset(test_encodings_en, test_encodings_zh)

    # Create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    evaluate_model_bleu(model, test_loader, tokenizer_zh)
