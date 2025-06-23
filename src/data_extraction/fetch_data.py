# src/data_extraction/fetch_data.py

import os, zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract():
    comp = "walmart-recruiting-store-sales-forecasting"
    # Корневая папка проекта — это текущая рабочая директория
    project_root = os.getcwd()
    out_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(out_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {comp} into {out_dir} …")
    api.competition_download_files(comp, path=out_dir, quiet=False)

    for fname in os.listdir(out_dir):
        if fname.endswith(".zip"):
            path = os.path.join(out_dir, fname)
            print(f"Extracting {path} …")
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(out_dir)
            os.remove(path)

if __name__ == "__main__":
    download_and_extract()
