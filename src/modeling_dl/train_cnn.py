# src/modeling_dl/train_cnn.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.optim import AdamW
from tqdm.auto import tqdm

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, num_filters=100, filter_sizes=(3,4,5), num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, emb_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [B, 1, L, E]
        conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # list of [B, F, L-fs+1]
        pooled = [nn.functional.max_pool1d(c, c.size(2)).squeeze(2) for c in conved]  # [B, F]
        cat = torch.cat(pooled, dim=1)        # [B, F*len(filter_sizes)]
        return self.fc(self.dropout(cat))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Загружаем токенизированный датасет
    ds = load_from_disk("data/reviews/tokenized")
    train_ds = ds["train"].with_format("torch", columns=["input_ids", "label"])
    val_ds   = ds["test"].with_format("torch", columns=["input_ids", "label"])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128)

    # 2) Используем токенизатор, чтобы узнать vocab_size
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # 3) Инициализируем модель, оптимизатор и критерий
    model = TextCNN(vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    # 4) Цикл обучения
    num_epochs = 3
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

        # 5) Валидация
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

    # 6) Сохраняем модель
    os.makedirs("models_dl", exist_ok=True)
    torch.save(model.state_dict(), "models_dl/textcnn.pt")
    print("Saved TextCNN to models_dl/textcnn.pt")

if __name__ == "__main__":
    train()
