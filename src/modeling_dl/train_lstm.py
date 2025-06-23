# src/modeling_dl/train_lstm.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm.auto import tqdm

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128, num_layers=1,
                 bidirectional=True, num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_multiplier, num_classes)

    def forward(self, x):
        # x: [B, L]
        emb = self.embedding(x)  # [B, L, E]
        outputs, (hn, cn) = self.lstm(emb)  
        # hn: [num_layers * num_directions, B, hidden_dim]
        if self.bidirectional:
            # concatenate forward and backward hidden states of last layer
            # forward state is hn[-2], backward is hn[-1]
            forward_hidden = hn[-2]
            backward_hidden = hn[-1]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            last_hidden = hn[-1]  # [B, hidden_dim]
        return self.fc(self.dropout(last_hidden))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load tokenized dataset
    ds = load_from_disk("data/reviews/tokenized")
    train_ds = ds["train"].with_format("torch", columns=["input_ids", "label"])
    val_ds   = ds["test"].with_format("torch",   columns=["input_ids", "label"])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128)

    # 2) Get vocab size from tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # 3) Initialize model, optimizer, loss
    model = TextLSTM(vocab_size=vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    # 4) Training loop
    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for batch in pbar:
            inputs = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # 5) Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                labels = batch["label"].to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        print(f"[Epoch {epoch}] Validation Accuracy: {acc:.4f}")

    # 6) Save model
    os.makedirs("models_dl", exist_ok=True)
    torch.save(model.state_dict(), "models_dl/textlstm.pt")
    print("Saved TextLSTM to models_dl/textlstm.pt")

if __name__ == "__main__":
    train()
