# src/data_extraction_dl/fetch_reviews.py
from datasets import load_dataset
import os

def fetch_reviews():
    os.makedirs("data/reviews", exist_ok=True)
    ds = load_dataset("amazon_polarity")
    ds["train"].to_json("data/reviews/train.jsonl", orient="records", lines=True)
    ds["test"].to_json("data/reviews/test.jsonl", orient="records", lines=True)
    print("Saved train/test to data/reviews/")
    
if __name__ == "__main__":
    fetch_reviews()