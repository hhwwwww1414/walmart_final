# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from wordcloud import WordCloud

# Попытаемся импортировать seasonal_decompose
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

# Настройка страницы
st.set_page_config(
    page_title="Walmart Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_sales():
    sales = pd.read_csv("../data/raw/train.csv", parse_dates=["Date"])
    feats = pd.read_csv("../data/raw/features.csv", parse_dates=["Date"])
    stores = pd.read_csv("../data/raw/stores.csv")
    df = (
        sales
        .merge(feats, on=["Store", "Date", "IsHoliday"], how="left")
        .merge(stores, on="Store", how="left")
    )
    return df

@st.cache_data
def load_reviews():
    try:
        df = pd.read_json("../data/reviews/train.jsonl", lines=True)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame({"review_text": []})
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols and "review_text" not in df.columns:
        df = df.rename(columns={text_cols[0]: "review_text"})
    df["review_text"] = df.get("review_text", "").astype(str)
    return df[["review_text"]]

# Загрузка данных
df = load_sales()
reviews = load_reviews()

# Sidebar: фильтры по дате, магазину, отделу
st.sidebar.title("Фильтры")
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Выберите период", (min_date, max_date),
    min_value=min_date, max_value=max_date
)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[df["Date"].between(start, end)]

stores = ["Все"] + sorted(df["Store"].unique().tolist())
sel_store = st.sidebar.selectbox("Магазин", stores)
if sel_store != "Все":
    df = df[df["Store"] == sel_store]

depts = ["Все"] + sorted(df["Dept"].unique().tolist())
sel_dept = st.sidebar.selectbox("Отдел", depts)
if sel_dept != "Все":
    df = df[df["Dept"] == sel_dept]

# Раздел 1: столбчатая диаграмма продаж
st.title("📊 Еженедельные продажи Walmart (столбцы)")
if "Weekly_Sales" in df.columns:
    agg = df.groupby(["Date", "IsHoliday"])["Weekly_Sales"].sum().reset_index()
    fig1 = px.bar(
        agg, x="Date", y="Weekly_Sales", color="IsHoliday",
        barmode="stack",
        labels={"Weekly_Sales": "Сумма продаж", "IsHoliday": "Праздник"},
        title="Продажи с выделением праздничных недель"
    )
    fig1.update_layout(xaxis_title="Дата", yaxis_title="Weekly Sales")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Нет `Weekly_Sales` для визуализации.")

# Раздел 2: сезонная декомпозиция
st.subheader("🔍 Сезонная декомпозиция суммарных продаж")
if _has_statsmodels and "Weekly_Sales" in df.columns:
    daily = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    if len(daily) >= 2*52:
        decomp = seasonal_decompose(
            daily.set_index("Date")["Weekly_Sales"],
            model="additive", period=52
        )
        comp = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Наблюдения", "Тренд", "Сезонность", "Остатки"]
        )
        comp.add_trace(go.Scatter(x=daily["Date"], y=daily["Weekly_Sales"]), row=1, col=1)
        comp.add_trace(go.Scatter(x=daily["Date"], y=decomp.trend), row=2, col=1)
        comp.add_trace(go.Scatter(x=daily["Date"], y=decomp.seasonal), row=3, col=1)
        comp.add_trace(go.Scatter(x=daily["Date"], y=decomp.resid), row=4, col=1)
        comp.update_layout(height=800, showlegend=False)
        st.plotly_chart(comp, use_container_width=True)
    else:
        st.info("Недостаточно данных для декомпозиции.")
else:
    st.info("Установите `statsmodels`, чтобы увидеть сезонную декомпозицию.")

# Раздел 3: корреляционная матрица
st.subheader("📈 Корреляционная матрица")
features = ["Weekly_Sales", "Size", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
avail = [f for f in features if f in df.columns]
sel_corr = st.multiselect("Выберите признаки", avail, default=avail)
if sel_corr:
    corr = df[sel_corr].corr()
    heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu", color_continuous_midpoint=0
    )
    heat.update_layout(height=500, margin=dict(t=50))
    st.plotly_chart(heat, use_container_width=True)
else:
    st.info("Выберите хотя бы один признак.")

# Раздел 4: scatter-плоты с управлением объёмом данных и целевой переменной
st.subheader("🔀 Scatter: зависимость признаков")
# 1. Выбор целевой переменной (любая числовая колонка)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
target = st.selectbox(
    "Выберите целевую переменную (ось Y)",
    numeric_cols,
    index=numeric_cols.index("Weekly_Sales") if "Weekly_Sales" in numeric_cols else 0
)

# 2. Выбор признаков для оси X
scatter_feats = st.multiselect(
    "Признаки для оси X",
    [c for c in numeric_cols if c != target],
    default=[c for c in numeric_cols if c not in ("Store","Dept","IsHoliday","Date",target)][:3]
)

# 3. Объём выборки
max_n = len(df)
options = [1000, 5000, 10000]
options = [o for o in options if o < max_n] + [max_n]
sample_n = st.select_slider(
    "Количество точек для отрисовки",
    options=options,
    value=min(5000, max_n)
)

if not scatter_feats:
    st.info("Выберите хотя бы один признак для X.")
else:
    # делаем случайную подвыборку
    df_smp = df.sample(n=sample_n, random_state=42)
    # выводим сетку по 3 графика в ряд
    cols = 3
    for i in range(0, len(scatter_feats), cols):
        row_feats = scatter_feats[i:i+cols]
        cols_ui = st.columns(len(row_feats))
        for feat, col_ui in zip(row_feats, cols_ui):
            fig = px.scatter(
                df_smp,
                x=feat, y=target,
                color="IsHoliday" if "IsHoliday" in df_smp.columns else None,
                labels={feat: feat, target: target, "IsHoliday": "Holiday"},
                title=f"{feat} ↔ {target} (n={sample_n})"
            )
            fig.update_traces(marker=dict(size=5, opacity=0.5))
            fig.update_layout(height=300, margin=dict(t=40, b=20))
            col_ui.plotly_chart(fig, use_container_width=True)

# Раздел 5: география
if {"Latitude", "Longitude"}.issubset(df.columns):
    st.subheader("🌎 Продажи по локациям")
    store_loc = (
        df.groupby("Store")[["Latitude","Longitude","Weekly_Sales"]]
        .agg({"Latitude":"first","Longitude":"first","Weekly_Sales":"sum"})
        .reset_index()
    )
    map_fig = px.scatter_mapbox(
        store_loc,
        lat="Latitude", lon="Longitude",
        size="Weekly_Sales", hover_name="Store",
        zoom=3, mapbox_style="carto-positron",
        title="Продажи по магазинам"
    )
    st.plotly_chart(map_fig, use_container_width=True)

# Раздел 6: анализ отзывов
st.title("💬 Анализ отзывов пользователей")
if not reviews.empty:
    text = " ".join(reviews["review_text"])
    wc = WordCloud(width=800, height=300, background_color="white").generate(text)
    st.image(wc.to_array(), use_container_width=True)

    reviews["length"] = reviews["review_text"].str.len()
    hist = px.histogram(
        reviews, x="length", nbins=50,
        title="Распределение длины отзывов"
    )
    hist.update_layout(xaxis_title="Длина", yaxis_title="Частота", height=400)
    st.plotly_chart(hist, use_container_width=True)
else:
    st.info("Нет отзывов для анализа.")
