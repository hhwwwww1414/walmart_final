# src/modeling_ml/tune_histgb.py

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import uniform, randint

def stamp():
    return time.strftime("%H:%M:%S")

def tune_histgb():
    # 1) Загрузка данных и препроцессора
    df = pd.read_csv(os.path.join("data","processed","processed.csv"))
    y = df.pop("Weekly_Sales")
    X = df

    preproc = joblib.load(os.path.join("models","preprocessor.joblib"))
    X_t = preproc.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    # 2) Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_t, y, test_size=0.2, random_state=42
    )

    # 3) Пул гиперпараметров
    param_dist = {
        "learning_rate": uniform(0.01, 0.3),    # от 0.01 до 0.31
        "max_iter": randint(50, 201),           # от 50 до 200 итераций
        "max_depth": randint(2, 8),             # глубина от 2 до 7
        "min_samples_leaf": randint(20, 101),   # min size листа
        "l2_regularization": uniform(0.0, 1.0)  # l2 от 0 до 1
    }

    model = HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)

    search = RandomizedSearchCV(
        estimator   = model,
        param_distributions = param_dist,
        n_iter      = 20,
        cv          = 3,
        scoring     = "neg_mean_absolute_error",
        n_jobs      = -1,
        verbose     = 2,
        random_state= 42
    )

    # 4) Запуск поиска
    print(f"[{stamp()}] Starting RandomizedSearchCV for HistGB …")
    start = time.time()
    search.fit(X_train, y_train)
    print(f"[{stamp()}] Done in {time.time()-start:.1f}s")
    print("Best params:", search.best_params_)

    # 5) Оценка на валидации
    best = search.best_estimator_
    preds = best.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"[{stamp()}] Validation MAE of best HistGB: {mae:.2f}")

    # 6) Сохраняем результаты
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_csv("results/histgb_tuning_results.csv", index=False)
    joblib.dump(best, "models/HistGB_best.joblib")

    print(f"[{stamp()}] Tuning results saved to results/histgb_tuning_results.csv")
    print(f"[{stamp()}] Best model saved to models/HistGB_best.joblib")

if __name__ == "__main__":
    tune_histgb()
