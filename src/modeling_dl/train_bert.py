# src/modeling_dl/train_bert.py

import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc      = evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"]
    prec     = evaluate.load("precision").compute(predictions=preds, references=labels)["precision"]
    rec      = evaluate.load("recall").compute(predictions=preds, references=labels)["recall"]
    f1       = evaluate.load("f1").compute(predictions=preds, references=labels)["f1"]
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def train_bert(model_name="bert-base-uncased", output_dir="models_dl/bert"):
    # 1) Загружаем токенизированный корпус
    ds = load_from_disk("data/reviews/tokenized")

    # 2) Берём маленькую подвыборку
    train_ds = ds["train"].shuffle(seed=42).select(range(5000))
    eval_ds  = ds["test"].shuffle(seed=42).select(range(1000))

    # 3) Инициализируем токенизатор и модель
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # 4) Аргументы обучения: 1 эпоха, batch 32, без промежуточного eval
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_dir="logs",
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        do_train=True,
        do_eval=False,              # отключаем eval во время тренировки
    )

    # 5) Создаём Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,      # Trainer.evaluate() всё равно выполнит одну финальную eval
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 6) Тренируем
    print(">>> Starting fine-tuning BERT on 5k examples (batch_size=32)…")
    trainer.train()

    # 7) Финальная оценка
    print(">>> Running final evaluation on 1k examples…")
    metrics = trainer.evaluate()
    print("Final evaluation metrics:", metrics)

    # 8) Сохраняем модель
    trainer.save_model(output_dir)
    print(f"Saved BERT model to {output_dir}")

if __name__ == "__main__":
    train_bert()
