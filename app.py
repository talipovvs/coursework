import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import silhouette_score, accuracy_score, r2_score
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Data Explorer", layout="wide")
sns.set_style("whitegrid")

# -------------------------------------------------------------------
# Страницы (имитация мультистраничного приложения)
# -------------------------------------------------------------------
page = st.sidebar.selectbox("Выберите страницу", ["Главная", "Analysis Results"])

# -------------------------------------------------------------------
# Загрузка данных (одинаково для всех страниц)
# -------------------------------------------------------------------
df = pd.read_csv("dftrain_clean.csv")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != "Overview"]

# ===========================
# ГЛАВНАЯ СТРАНИЦА
# ===========================
if page == "Главная":
    st.title("Главная — Визуализация данных")

    st.subheader("Первые 20 строк датасета")
    st.dataframe(df.head(20))

    # KPI
    st.header("KPI и базовые статистики")
    col1, col2, col3 = st.columns(3)
    col1.metric("Общее количество записей", df.shape[0])
    col2.metric("Количество колонок", df.shape[1])
    col3.metric("Пропущенные значения", df.isna().sum().sum())

    # Базовая статистика
    st.subheader("Статистика числовых признаков")
    stats = df.describe().T[['mean', '50%', 'std']].rename(columns={'50%': 'median'})
    st.dataframe(stats)

    # Числовые распределения
    st.header("Распределение числовых признаков")
    for col in num_cols:
        st.write(f"### {col}")
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f"Гистограмма {col}")
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot {col}")
        st.pyplot(fig)

    # Категориальные признаки
    st.header("Категориальные признаки")
    for col in cat_cols:
        st.write(f"### {col}")
        vc = df[col].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(10, 4))
        vc.plot(kind='bar', ax=ax)
        ax.set_title(f"Bar chart (Top 20) — {col}")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        vc.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Pie chart — {col}")
        st.pyplot(fig)

    # Корреляционная матрица
    st.header("Корреляционная матрица")
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.info("Недостаточно числовых признаков для корреляционной матрицы")

# ===========================
# ANALYSIS RESULTS — Страница 2
# ===========================
elif page == "Analysis Results":
    st.title("Analysis Results — Результаты анализа")

    # ---------------------------
    # 1. КЛАСТЕРИЗАЦИЯ
    # ---------------------------
    st.header("Кластеры фильмов")
    cluster_features = ['IMDB_Rating', 'Gross', 'Runtime', 'No_of_Votes', 'Released_Year']
    missing = [c for c in cluster_features if c not in df.columns]
    if missing:
        st.error(f"Признаки для кластеризации отсутствуют: {missing}")
    else:
        df_num = df[cluster_features].copy().dropna()
        df_num['Gross_log'] = np.log1p(df_num['Gross'])
        df_num['Votes_log'] = np.log1p(df_num['No_of_Votes'])
        X = df_num[['IMDB_Rating', 'Gross_log', 'Runtime', 'Votes_log', 'Released_Year']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider("Количество кластеров", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df_num['Cluster'] = kmeans.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=42)
        pca_res = pca.fit_transform(X_scaled)
        df_num['PCA1'] = pca_res[:, 0]
        df_num['PCA2'] = pca_res[:, 1]

        fig = px.scatter(
            df_num,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            size="Gross_log",
            opacity=0.75,
            hover_data=["IMDB_Rating", "Gross", "Runtime", "No_of_Votes", "Released_Year"],
            title="Кластеры фильмов (PCA 2D)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 2. ТРЕНДЫ И ВРЕМЕННЫЕ РЯДЫ
    # ---------------------------
    st.header("Тренды и временные ряды")
    df_ts = df.copy()
    df_ts["date"] = pd.to_datetime(df_ts["Released_Year"], format="%Y", errors="coerce")
    df_ts = df_ts.dropna(subset=["date"]).sort_values("date")

    if df_ts is not None:
        num_cols_trend = [c for c in num_cols if c != "Released_Year"]
        if num_cols_trend:
            y_col = st.selectbox("Выберите числовой признак для тренда:", num_cols_trend)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_ts["date"], df_ts[y_col], linewidth=2)
            ax.set_title(f"Тренд: {y_col}")
            ax.set_xlabel("Дата")
            ax.set_ylabel(y_col)
            st.pyplot(fig)
        else:
            st.info("Нет числовых признаков для построения тренда.")

    # ---------------------------
    # 3. ROC-кривая
    # ---------------------------
    st.header("ROC-кривая")
    threshold = st.slider("Выберите порог для 'хорошего' фильма (IMDB_Rating):", 0.0, 10.0, 7.0)
    df['target'] = (df['IMDB_Rating'] >= threshold).astype(int)

    if 'Meta_score' in df.columns:
        df_plot = df.dropna(subset=['Meta_score'])
        y_true = df_plot['target']
        y_score = df_plot['Meta_score']

        if len(set(y_true)) < 2:
            st.info("Невозможно построить ROC — все фильмы относятся к одному классу при текущем пороге.")
        else:
            fpr, tpr, thr = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0,1], [0,1], '--', color='gray', label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC-кривая (порог IMDB >= {threshold})")
            ax.legend()
            st.write("Количество положительных фильмов:", y_true.sum())
            st.write("Количество отрицательных фильмов:", len(y_true) - y_true.sum())
            st.pyplot(fig)
    else:
        st.info("Колонка 'Meta_score' отсутствует — ROC-кривую построить невозможно.")


    st.header(" Ключевые метрики и сравнение моделей")

    # Silhouette Score
    sil_score = silhouette_score(X_scaled, df_num['Cluster']) if 'Cluster' in df_num.columns else None

    threshold = st.slider("Порог IMDB для 'хорошего' фильма", 0.0, 10.0, 8.3)
    df['target'] = (df['IMDB_Rating'] >= threshold).astype(int)

    # Делаем простую модель с Votes_log и Gross_log
    df['Gross_log'] = np.log1p(df['Gross'])
    df['Votes_log'] = np.log1p(df['No_of_Votes'])

    # Простая линейная комбинация для демо
    y_score = 0.3*df['Gross_log'] + 0.7*df['Votes_log']
    y_score = (y_score >= y_score.median()).astype(int)

    acc = accuracy_score(df['target'], y_score)
    st.metric("Accuracy (демо)", f"{acc:.3f}")

    # R² (демо регрессия)
    df_reg = df.copy()
    df_reg['Gross_log'] = np.log1p(df_reg['Gross'])
    df_reg['Votes_log'] = np.log1p(df_reg['No_of_Votes'])

    # Заполняем пропуски медианой
    df_reg['Gross_log'].fillna(df_reg['Gross_log'].median(), inplace=True)
    df_reg['Votes_log'].fillna(df_reg['Votes_log'].median(), inplace=True)
    df_reg['IMDB_Rating'].fillna(df_reg['IMDB_Rating'].median(), inplace=True)
    df_reg['Meta_score'].fillna(df_reg['Meta_score'].median(), inplace=True)

    X_reg = df_reg[['IMDB_Rating','Gross_log','Votes_log']]
    y_reg = df_reg['Meta_score']

    model = LinearRegression()
    model.fit(X_reg, y_reg)
    y_pred = model.predict(X_reg)
    r2 = r2_score(y_reg, y_pred)
    st.metric("R² (регрессия)", f"{r2:.3f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score (кластеризация)", f"{sil_score:.3f}" if sil_score else "N/A")
    col2.metric("Accuracy (демо)", f"{acc:.3f}")
    col3.metric("R² (демо регрессия)", f"{r2:.3f}" if r2 else "N/A")

    # Сравнительная таблица
    results = {
        "Модель": ["KMeans", "Бинарная классификация по медиане IMDB", "Простая регрессия по Meta_score"],
        "Метрика": ["Silhouette Score", "Accuracy", "R²"],
        "Значение": [sil_score, acc, r2]
    }
    df_results = pd.DataFrame(results)
    st.subheader("Сравнение моделей")
    st.dataframe(df_results)

    # Визуализация KPI
    st.subheader("Визуализация ключевых метрик")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(df_results["Модель"], df_results["Значение"], color=['skyblue','orange','green'])
    ax.set_ylabel("Значение")
    ax.set_title("Сравнение ключевых метрик моделей")
    for i, v in enumerate(df_results["Значение"]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)
