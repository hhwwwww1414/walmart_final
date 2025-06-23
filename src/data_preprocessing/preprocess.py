# src/data_preprocessing/preprocess.py

import os
import pandas as pd

def preprocess():
    import os
    import pandas as pd

    project_root = os.getcwd()
    raw_dir = os.path.join(project_root, "data", "raw")
    proc_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(raw_dir, "train.csv"), parse_dates=["Date"])
    features = pd.read_csv(os.path.join(raw_dir, "features.csv"), parse_dates=["Date"])
    stores = pd.read_csv(os.path.join(raw_dir, "stores.csv"))

    # Слияние
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left") \
              .merge(stores, on="Store", how="left")

    # Заполняем пропуски в Markdown-фичах нулями
    mark_cols = [f"MarkDown{i}" for i in range(1, 6)]
    df[mark_cols] = df[mark_cols].fillna(0)

    # Генерация временных признаков
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    df["DayOfWeek"]  = df["Date"].dt.dayofweek

    # Выбираем фичи и таргет
    target = "Weekly_Sales"
    # убираем Date и IsHoliday — оставим Holiday как булевую
    features = ["Store", "Dept", "Type", "Size", "Temperature", "Fuel_Price",
                "CPI", "Unemployment", "IsHoliday", "Year", "Month",
                "WeekOfYear", "DayOfWeek"] + mark_cols

    df_final = df[features + [target]].copy()

    # Сохраняем
    out_path = os.path.join(proc_dir, "processed.csv")
    df_final.to_csv(out_path, index=False)
    print(f"Processed data saved to {out_path}")

if __name__ == "__main__":
    preprocess()
