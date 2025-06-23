# src/data_preprocessing_dl/tokenize.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_and_save(model_name="bert-base-uncased", max_length=128):
    # 1) Загружаем корпус
    ds = load_dataset("amazon_polarity")
    # 2) Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3) Функция преобразования
    def preprocess(examples):
        return tokenizer(
            examples["content"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # 4) Применяем batched
    tokenized = ds.map(preprocess, batched=True, remove_columns=["content"])
    # 5) Сохраняем на диск
    os.makedirs("data/reviews/tokenized", exist_ok=True)
    tokenized.save_to_disk("data/reviews/tokenized")
    print("Tokenized dataset saved to data/reviews/tokenized")

if __name__ == "__main__":
    tokenize_and_save()
