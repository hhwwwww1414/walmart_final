# Прогнозирование продаж и анализ тональности

> Комплексный проект: прогнозирование объёмов розничных продаж и классификация отзывов с помощью классических ML- и Deep Learning-методов.

---

## Оглавление

- [Обзор](#обзор)  
- [Структура проекта](#структура-проекта)  
- [Установка и настройка](#установка-и-настройка)  
- [Данные](#данные)  
- [Классический ML-пайплайн](#классический-ml-пайплайн)  
  - [Результаты](#результаты-классического-ml)  
- [Deep Learning-пайплайн](#deep-learning-пайплайн)  
  - [Результаты](#результаты-deep-learning)  
- [Использование в Google Colab](#использование-в-google-colab)  
- [Визуализации](#дешборд)
---

## Обзор

1. **Прогнозирование продаж**  
   - **Кейс**: предсказывать еженедельные объёмы продаж по категориям товаров для оптимизации запасов.  
   - **Данные**: «Walmart Sales Forecasting» (Kaggle).  

2. **Анализ тональности**  
   - **Кейс**: автоматическая классификация отзывов (положительные/отрицательные) для поддержки принятия решений о товаре.  
   - **Данные**: «Amazon Polarity» (Hugging Face Datasets).  

Проект охватывает этапы EDA, предобработки, обучения, настройки гиперпараметров и итогового сравнения моделей.

---

## Структура проекта

```
walmart-forecasting/
├── data/                      # исходные и обработанные данные
│   ├── raw/                   # (игнорируется) оригинальные CSV
│   └── processed/             # очищенные CSV
├── src/                       # исходный код
│   ├── data_extraction/       # извлечение данных по продажам
│   ├── data_preprocessing/    # предобработка продаж
│   ├── modeling_ml/           # классические ML-скрипты
│   ├── data_extraction_dl/    # извлечение отзывов
│   ├── data_preprocessing_dl/ # токенизация отзывов
│   └── modeling_dl/           # DL-модели и сравнение
├── notebooks/                 # ноутбуки EDA
│   ├── 01_EDA.ipynb           # анализ продаж
│   └── 02_DL_EDA.ipynb        # анализ отзывов
├── requirements.txt           # зависимости Python
└── README.md                  # этот файл
```

---

## Установка и настройка

```bash
git clone https://github.com/hhwwwww1414/walmart-forecasting.git
cd walmart-forecasting
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Данные

- **Walmart Sales Forecasting**  
  Еженедельные продажи магазинов и отделов с дополнительными признаками: температура, цены на топливо, CPI, безработица, акции, праздники.

- **Amazon Polarity**  
  3.6 млн тренировочных и 400 тыс тестовых отзывов, метки 0/1.

### Скрипты

```bash
# Продажи
python -m src.data_extraction.fetch_data
python -m src.data_preprocessing.preprocess

# Отзывы
python src/data_extraction_dl/fetch_reviews.py
python -m src.data_preprocessing_dl/tokenize
```

---

## Классический ML-пайплайн

1. Препроцессинг: StandardScaler, PCA, OneHotEncoder.  
2. Модели: LinearRegression, DecisionTree, RandomForest, HistGradientBoosting, SVR, MLP.  
3. Тюнинг: Grid/RandomSearch для HistGradientBoosting.  
4. Финальное обучение HistGB на полном наборе.

### Результаты классического ML

| Модель                      | MAE    | RMSE    | R²    |
|-----------------------------|--------|---------|-------|
| LinearRegression            | 8 162.9|13 355.9 |0.658  |
| DecisionTree (max_depth=10) | 9 219.5|13 640.5 |0.643  |
| RandomForest                | 9 077.4|13 354.9 |0.658  |
| HistGradientBoosting        | 8 365.9|12 626.9 |0.694  |
| SVR (linear)                | 8 329.8|13 852.0 |0.632  |

> **Итоговая модель**: `HistGradientBoostingRegressor` → MAE≈8 366, R²≈0.694

---

## Deep Learning-пайплайн

1. EDA: анализ распределения длин и баланса классов.  
2. Токенизация: `bert-base-uncased`, `max_length=128`.  
3. Модели: TextCNN, TextLSTM (BiLSTM), BERT (fine-tuning).  
4. Сравнение на сэмпле теста (10 000 отзывов).

### Результаты Deep Learning

| Модель   | Acc   | Prec_neg | Recall_neg | F1_neg | Prec_pos | Recall_pos | F1_pos |
|----------|-------|----------|------------|--------|----------|------------|--------|
| TextCNN  |0.9241 |0.9173    |0.9308      |0.9240  |0.9310    |0.9175      |0.9242  |
| TextLSTM |0.9215 |0.9108    |0.9330      |0.9218  |0.9325    |0.9102      |0.9212  |
| **BERT** |0.8931 |0.8484    |0.9550      |0.8986  |0.9495    |0.8322      |0.8870  |

> *BERT обучен на выборке 5 000 / 1 000 примеров; полная дообучка на GPU даёт лучшие результаты.*

---

## Использование в Google Colab

```bash
!git clone https://github.com/hhwwwww1414/walmart-forecasting.git
%cd walmart-forecasting
!pip install -r requirements.txt

# Запуск пайплайна
!python -m src.data_extraction.fetch_data
!python -m src.data_preprocessing.preprocess
!python src/data_extraction_dl/fetch_reviews.py
!python -m src.data_preprocessing_dl.tokenize
!python src/modeling_ml/train_models.py
!python src/modeling_ml/tune_histgb.py
!python src/modeling_ml/final_train.py
!python src/modeling_dl/train_cnn.py
!python src/modeling_dl/train_lstm.py
!python -m src.modeling_dl.train_bert
!python -m src.modeling_dl.compare_models
```

---
## Дешборд

- [Дешборд с нашими результатами](https://walmartvisual-mfdejtf2dpfxhdjqppsipw.streamlit.app/)
