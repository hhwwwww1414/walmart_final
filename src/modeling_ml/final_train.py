# src/modeling_ml/final_train.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

def final_train():
    # 1) Загрузка обработанного датасета
    df = pd.read_csv(os.path.join("data","processed","processed.csv"))
    y = df.pop("Weekly_Sales")
    X = df

    # 2) Подгрузка препроцессора и трансформация всего X
    preproc = joblib.load(os.path.join("models","preprocessor.joblib"))
    X_t = preproc.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    # 3) Настройка модели с лучшими гиперпараметрами
    best_params = {
        "learning_rate": 0.2106523759,
        "max_iter": 148,
        "max_depth": 6,
        "min_samples_leaf": 63,
        "l2_regularization": 0.3708182524,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1
    }
    model = HistGradientBoostingRegressor(**best_params)

    # 4) Обучение на всём датасете
    print("Training final HistGB on full data…")
    model.fit(X_t, y)

    # 5) Сохранение итоговой модели
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/HistGB_final.joblib")
    print("Final model saved to models/HistGB_final.joblib")

if __name__ == "__main__":
    final_train()
