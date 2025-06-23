# src/modeling_ml/train_models.py

import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def stamp():
    return time.strftime("%H:%M:%S")

def train_and_evaluate():
    # 1) Загружаем препроцессированный CSV
    path = os.path.join(os.getcwd(), "data", "processed", "processed.csv")
    df = pd.read_csv(path)

    # 2) X/y split
    y = df.pop("Weekly_Sales")
    X = df

    # 3) Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Фабрика признаков (числовые + OHE)
    num_feats = [
        "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "Year", "Month", "WeekOfYear", "DayOfWeek"
    ] + [f"MarkDown{i}" for i in range(1, 6)]
    cat_feats = ["Store", "Dept", "Type", "IsHoliday"]

    num_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42))
    ])
    cat_pipe = OneHotEncoder(handle_unknown="ignore")

    preproc = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ], sparse_threshold=0.0)

    print(f"[{stamp()}] Fitting preprocessor…")
    X_tr_t = preproc.fit_transform(X_train)
    X_vl_t = preproc.transform(X_val)

    # Если это sparse, конвертим в плотный
    if hasattr(X_tr_t, "toarray"):
        X_tr_t = X_tr_t.toarray()
        X_vl_t = X_vl_t.toarray()

    os.makedirs("models", exist_ok=True)
    joblib.dump(preproc, "models/preprocessor.joblib")

    # 5) Готовим подвыборку для MLP (10%)
    mlp_frac = 0.1
    idx = np.random.choice(len(X_tr_t), int(len(X_tr_t)*mlp_frac), replace=False)
    X_tr_mlp = X_tr_t[idx]
    y_tr_mlp = y_train.iloc[idx]

    # 6) Словарь моделей
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=20, max_depth=10, n_jobs=-1, random_state=42
        ),
        "HistGB": HistGradientBoostingRegressor(
            max_iter=100, max_depth=3, random_state=42
        ),
        "MLP": MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
    }

    results = []
    os.makedirs("results", exist_ok=True)

    # 7) Обучение и оценка
    for name, model in models.items():
        print(f"[{stamp()}] Training {name}…")
        start = time.time()

        if name == "MLP":
            model.fit(X_tr_mlp, y_tr_mlp)
        else:
            model.fit(X_tr_t, y_train)

        duration = time.time() - start
        preds = model.predict(X_vl_t)

        mae  = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2   = r2_score(y_val, preds)

        joblib.dump(model, f"models/{name}.joblib")
        results.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "time_s": duration
        })

        print(f"   → {name}: MAE={mae:.1f}, RMSE={rmse:.1f}, R2={r2:.3f} (took {duration:.1f}s)")

    # 8) Сохраняем результаты
    df_res = pd.DataFrame(results).sort_values("mae")
    df_res.to_csv("results/classical_ml_results.csv", index=False)
    print(f"[{stamp()}] Done! Results in results/classical_ml_results.csv")

if __name__ == "__main__":
    train_and_evaluate()
