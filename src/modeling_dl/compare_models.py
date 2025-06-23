# src/modeling_dl/compare_models.py

import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from tqdm.auto import tqdm

# Убедитесь, что эти модули импортируются из пакетов:
from src.modeling_dl.train_cnn  import TextCNN
from src.modeling_dl.train_lstm import TextLSTM

def eval_pytorch_model(model, path, test_ds, device="cpu", batch_size=256, name="Model"):
    """Оценка PyTorch-модели с прогресс-баром."""
    print(f"[{time.strftime('%H:%M:%S')}] Loading weights for {name}...")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()

    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=(device != "cpu")
    )
    all_preds, all_labels = [], []

    print(f"[{time.strftime('%H:%M:%S')}] Running inference for {name} ({len(test_ds)} samples)...")
    for batch in tqdm(loader, desc=f"{name} batches"):
        inputs = batch["input_ids"].to(device)
        labels = batch["label"].numpy()
        with torch.no_grad():
            logits = model(inputs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1], zero_division=0
    )
    print(f"[{time.strftime('%H:%M:%S')}] {name} done: acc={acc:.4f}")
    return acc, (prec, rec, f1)

def eval_bert_model(model_dir, test_ds):
    """Быстрая оценка BERT через Trainer.predict."""
    print(f"[{time.strftime('%H:%M:%S')}] Loading BERT from {model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    trainer = Trainer(model=model, tokenizer=tokenizer)

    print(f"[{time.strftime('%H:%M:%S')}] Running BERT.predict on {len(test_ds)} samples...")
    preds_output = trainer.predict(test_ds)

    logits = preds_output.predictions
    y_true = preds_output.label_ids
    y_pred = np.argmax(logits, axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1], zero_division=0
    )
    print(f"[{time.strftime('%H:%M:%S')}] BERT done: acc={acc:.4f}")
    return acc, (prec, rec, f1)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] Device: {device}")

    # 1) Загружаем и делаем подвыборку
    print(f"[{time.strftime('%H:%M:%S')}] Loading tokenized test set and sampling 10000 ...")
    ds = load_from_disk("data/reviews/tokenized")
    test_small = (
        ds["test"]
        .shuffle(seed=42)
        .select(range(10000))
        .with_format("torch", columns=["input_ids","label"])
    )

    # 2) Вытаскиваем vocab_size для инициализации CNN/LSTM
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # 3) Оценка TextCNN
    acc_cnn, (p_cnn, r_cnn, f_cnn) = eval_pytorch_model(
        TextCNN(vocab_size),
        "models_dl/textcnn.pt",
        test_small,
        device=device,
        batch_size=256,
        name="TextCNN"
    )

    # 4) Оценка TextLSTM
    acc_lstm, (p_lstm, r_lstm, f_lstm) = eval_pytorch_model(
        TextLSTM(vocab_size=vocab_size),
        "models_dl/textlstm.pt",
        test_small,
        device=device,
        batch_size=256,
        name="TextLSTM"
    )

    # 5) Оценка BERT
    acc_bert, (p_bert, r_bert, f_bert) = eval_bert_model("models_dl/bert", test_small)

    # 6) Сбор результатов в DataFrame
    rows = [
        {
            "model": "TextCNN",
            "accuracy": acc_cnn,
            "precision_neg": p_cnn[0], "recall_neg": r_cnn[0], "f1_neg": f_cnn[0],
            "precision_pos": p_cnn[1], "recall_pos": r_cnn[1], "f1_pos": f_cnn[1],
        },
        {
            "model": "TextLSTM",
            "accuracy": acc_lstm,
            "precision_neg": p_lstm[0], "recall_neg": r_lstm[0], "f1_neg": f_lstm[0],
            "precision_pos": p_lstm[1], "recall_pos": r_lstm[1], "f1_pos": f_lstm[1],
        },
        {
            "model": "BERT",
            "accuracy": acc_bert,
            "precision_neg": p_bert[0], "recall_neg": r_bert[0], "f1_neg": f_bert[0],
            "precision_pos": p_bert[1], "recall_pos": r_bert[1], "f1_pos": f_bert[1],
        }
    ]
    df = pd.DataFrame(rows)

    # 7) Сохранение результатов
    os.makedirs("results_dl", exist_ok=True)
    df.to_csv("results_dl/dl_comparison.csv", index=False)
    print(f"[{time.strftime('%H:%M:%S')}] Saved results to results_dl/dl_comparison.csv\n")
    print(df)

if __name__ == "__main__":
    main()
