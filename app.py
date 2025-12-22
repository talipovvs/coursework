import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, roc_auc_score, silhouette_score, accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Data Explorer", layout="wide")
sns.set_style("whitegrid")

# -------------------------------------------------------------------
# Страницы
# -------------------------------------------------------------------
page = st.sidebar.selectbox("Выберите страницу", ["Главная", "Analysis Results"])

# -------------------------------------------------------------------
# Загрузка данных
# -------------------------------------------------------------------
df = pd.read_csv("dftrain_clean.csv")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != "Overview"]

FIG_SIZE = (7, 4)

# ===========================
# ГЛАВНАЯ СТРАНИЦА
# ===========================
if page == "Главная":
    st.title("Главная — Визуализация данных")

    st.subheader("📘 Описание колонок датасета")

    columns_description = {
        "Series_Title": "Название фильма",
        "Released_Year": "Год выхода фильма",
        "Runtime": "Длительность фильма (в минутах)",
        "Genre": "Жанр фильма",
        "IMDB_Rating": "Рейтинг IMDb (оценка зрителей)",
        "Meta_score": "Оценка критиков (Metacritic)",
        "Director": "Режиссёр фильма",
        "Star1": "Главная звезда фильма",
        "Star2": "Вторая по значимости звезда",
        "Star3": "Третья звезда фильма",
        "Star4": "Четвёртая звезда фильма",
        "No_of_Votes": "Количество голосов на IMDb",
        "Gross": "Кассовые сборы фильма (USD)",
        "Overview": "Краткое описание фильма"
    }

    df_columns_info = pd.DataFrame({
        "Название колонки": columns_description.keys(),
        "Описание": columns_description.values()
    })

    st.dataframe(df_columns_info, use_container_width=True)

    st.subheader("Первые 20 строк датасета")
    st.dataframe(df.head(20))

    st.header("KPI и базовые статистики")
    col1, col2, col3 = st.columns(3)
    col1.metric("Общее количество записей", df.shape[0])
    col2.metric("Количество колонок", df.shape[1])
    col3.metric("Пропущенные значения", df.isna().sum().sum())

    st.subheader("Статистика числовых признаков")
    stats = df.describe().T[['mean', '50%', 'std']].rename(columns={'50%': 'median'})
    st.dataframe(stats.style.format("{:.2f}"))

    st.header("Распределение числовых признаков (интерактивно)")
    for col in num_cols:
        st.write(f"### {col}")
        fig = px.histogram(df, x=col, nbins=50, marginal="box", hover_data=[col])
        fig.update_layout(title=f"Гистограмма и Boxplot — {col}", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.header("Категориальные признаки (интерактивно)")
    TOP_N = 20  # показывать только топ N значений

    for col in cat_cols:
        st.write(f"### {col}")

        # Берем топ-N значений
        vc = df[col].value_counts().head(TOP_N).reset_index()
        vc.columns = [col, "count"]

        # Горизонтальный bar chart
        fig = px.bar(
            vc,
            x="count",
            y=col,
            orientation='h',
            text="count",
            hover_data=[col, "count"]
        )
        fig.update_layout(
            title=f"Bar chart — топ {TOP_N} значений {col}",
            yaxis={'categoryorder':'total ascending'},  # сортировка по убыванию
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Pie chart (только для наглядности)
    fig_pie = px.pie(vc, names=col, values="count", hover_data=[col, "count"], hole=0.3)
    fig_pie.update_layout(title=f"Pie chart — топ {TOP_N} значений {col}", height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.header("Корреляционная матрица (интерактивно)")
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(title="Correlation Heatmap", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Недостаточно числовых признаков для корреляционной матрицы")

    # Кнопка для обновления данных (пример)
    if st.button("Обновить данные"):
        st.experimental_rerun()


# ===========================
# ANALYSIS RESULTS
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
        df_num[['PCA1','PCA2']] = pca.fit_transform(X_scaled)

        fig = px.scatter(
            df_num, x="PCA1", y="PCA2", color="Cluster", size="Gross_log", opacity=0.75,
            hover_data=["IMDB_Rating", "Gross", "Runtime", "No_of_Votes", "Released_Year"],
            title="Кластеры фильмов (PCA 2D)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 2. ТРЕНДЫ И ВРЕМЕННЫЕ РЯДЫ
    # ---------------------------
    st.header("Тренды и временные ряды")

    df_ts = df.copy()
    df_ts['Released_Year'] = pd.to_numeric(df_ts['Released_Year'], errors='coerce')
    df_ts['Runtime'] = pd.to_numeric(df_ts['Runtime'], errors='coerce')

    # Выбираем числовой признак для тренда
    num_cols_trend = [c for c in num_cols if c != "Released_Year"]
    if num_cols_trend:
        y_col = st.selectbox("Выберите числовой признак для тренда:", num_cols_trend)

        # Убираем строки с пропущенными значениями в Released_Year и выбранной колонке
        df_ts_clean = df_ts.dropna(subset=['Released_Year', y_col])

        # Агрегируем по году
        df_yearly = df_ts_clean.groupby('Released_Year')[y_col].mean().reset_index()
        df_yearly.rename(columns={'Released_Year': 'Year'}, inplace=True)

        # Строим график
        fig = px.line(df_yearly, x='Year', y=y_col, title=f"Тренд: {y_col}", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет числовых признаков для построения тренда.")


    # ---------------------------
    # 3. ROC-кривая
    # ---------------------------
    st.header("ROC-кривая")
    threshold = st.slider("Порог IMDB для 'хорошего' фильма", 0.0, 10.0, 7.0)
    df['target'] = (df['IMDB_Rating'] >= threshold).astype(int)
    if 'Meta_score' in df.columns:
        df_plot = df.dropna(subset=['Meta_score'])
        y_true = df_plot['target']
        y_score = df_plot['Meta_score']
        if len(set(y_true)) < 2:
            st.info("Невозможно построить ROC — все фильмы одного класса.")
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            fig = px.area(
                x=fpr, y=tpr, title=f"ROC-кривая (AUC={auc:.3f})",
                labels=dict(x="False Positive Rate", y="True Positive Rate")
            )
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 4. МОДЕЛИ И KPI
    # ---------------------------
    st.header("Ключевые метрики и сравнение моделей")
    sil_score = silhouette_score(X_scaled, df_num['Cluster']) if 'Cluster' in df_num.columns else None
    df['Gross_log'] = np.log1p(df['Gross'])
    df['Votes_log'] = np.log1p(df['No_of_Votes'])
    y_score_demo = (0.3*df['Gross_log'] + 0.7*df['Votes_log'] >= df['Gross_log'].median()).astype(int)
    acc = accuracy_score(df['target'], y_score_demo)

    # Простая регрессия
    df_reg = df.copy()
    for c in ['Gross_log','Votes_log','IMDB_Rating','Meta_score']:
        df_reg[c].fillna(df_reg[c].median(), inplace=True)
    X_reg = df_reg[['IMDB_Rating','Gross_log','Votes_log']]
    y_reg = df_reg['Meta_score']
    model = LinearRegression()
    model.fit(X_reg, y_reg)
    y_pred = model.predict(X_reg)
    r2 = r2_score(y_reg, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score (кластеризация)", f"{sil_score:.3f}" if sil_score else "N/A")
    col2.metric("Accuracy (демо)", f"{acc:.3f}")
    col3.metric("R² (регрессия)", f"{r2:.3f}")

    results = pd.DataFrame({
        "Модель":["KMeans","Бинарная классификация","Линейная регрессия"],
        "Метрика":["Silhouette Score","Accuracy","R²"],
        "Значение":[sil_score, acc, r2]
    })
    st.dataframe(results)

    fig = px.bar(results, x="Модель", y="Значение", color="Метрика", text="Значение")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 5. Insights
    # ---------------------------
    st.header("Insights по кластерам")
    if 'Cluster' in df_num.columns:
        selected_cluster = st.selectbox("Выберите кластер", sorted(df_num['Cluster'].unique()))
        cluster_df = df_num[df_num['Cluster']==selected_cluster]
        global_mean = df_num[['Gross','IMDB_Rating','No_of_Votes']].mean()
        cluster_mean = cluster_df[['Gross','IMDB_Rating','No_of_Votes']].mean()
        diff_pct = ((cluster_mean-global_mean)/global_mean)*100

        col1, col2, col3 = st.columns(3)
        col1.metric("Средний Gross", f"{cluster_mean['Gross']:.0f}", f"{diff_pct['Gross']:.1f}%")
        col2.metric("Средний IMDB", f"{cluster_mean['IMDB_Rating']:.2f}", f"{diff_pct['IMDB_Rating']:.1f}%")
        col3.metric("Среднее число голосов", f"{cluster_mean['No_of_Votes']:.0f}", f"{diff_pct['No_of_Votes']:.1f}%")
        st.markdown(f"Фильмы кластера **{selected_cluster}** имеют доход {'выше' if diff_pct['Gross']>0 else 'ниже'} среднего на {abs(diff_pct['Gross']):.1f}%, рейтинг {'выше' if diff_pct['IMDB_Rating']>0 else 'ниже'} среднего на {abs(diff_pct['IMDB_Rating']):.1f}%")

    # ---------------------------
    # 6. Предсказания модели
    # ---------------------------
    st.header("Предсказания модели (Meta_score)")
    df_pred = df_reg.copy()
    df_pred['Predicted_Meta'] = y_pred
    fig = px.line(
        df_pred.head(100), y=['Meta_score','Predicted_Meta'],
        labels={"value":"Meta_score","index":"Порядок фильмов"},
        title="Факт vs Предсказание (первые 100 фильмов)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 7. Feature importance
    # ---------------------------
    st.header("Влияние признаков (коэффициенты линейной регрессии)")
    coef_df = pd.DataFrame({"Feature":X_reg.columns,"Coefficient":model.coef_}).set_index("Feature")
    fig = px.imshow(coef_df, text_auto=True, color_continuous_scale='RdBu', title="Коэффициенты линейной регрессии")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # 8. Сценарии анализа
    # ---------------------------
    st.header("Сценарии анализа")
    scenario = st.radio("Выберите сценарий", ["Оценка по голосам","Оценка по доходу","Комбинированный"])
    if scenario=="Оценка по голосам": score=df_reg['Votes_log']
    elif scenario=="Оценка по доходу": score=df_reg['Gross_log']
    else: score=0.5*df_reg['Votes_log']+0.5*df_reg['Gross_log']
    fig = px.histogram(score, nbins=30, title=f"Распределение оценки — {scenario}")
    st.plotly_chart(fig, use_container_width=True)

    
    st.header("Влияние факторов на Meta_score")

    importance = pd.DataFrame({
        "Фактор": X_reg.columns,
        "Влияние": model.coef_
    }).sort_values("Влияние", ascending=False)

    fig = px.bar(
        importance,
        x="Влияние",
        y="Фактор",
        orientation="h",
        title="Влияние признаков на прогноз Meta_score"
    )

    st.plotly_chart(fig, use_container_width=True)

  
    # ---------------------------
    # 10. Актёры-катализаторы успеха
    # ---------------------------
    st.header("Актёры и влияние на успех фильма")

    st.markdown("""
    Actor Impact Score показывает, насколько фильмы с участием актёра
    в среднем зарабатывают больше (или меньше), чем рынок в целом.
    """)

    # Разделяем список актёров
    actors_df = df[['Star1', 'Star2', 'Star3', 'Star4', 'Gross']].dropna()

    actors_long = actors_df.melt(
        id_vars='Gross',
        value_vars=['Star1', 'Star2', 'Star3', 'Star4'],
        value_name='Actor'
    )

    actor_stats = (
        actors_long
        .groupby('Actor')
        .agg(
            Avg_Gross=('Gross', 'mean'),
            Movies=('Gross', 'count')
        )
        .query("Movies >= 5")  # фильтр по количеству фильмов
    )

    global_avg = df['Gross'].mean()
    actor_stats['Impact'] = actor_stats['Avg_Gross'] - global_avg

    top_actors = actor_stats.sort_values('Impact', ascending=False).head(15).reset_index()

    fig = px.bar(
        top_actors,
        x='Impact',
        y='Actor',
        orientation='h',
        text='Movies',
        title="Актёры-катализаторы коммерческого успеха",
        hover_data=['Avg_Gross', 'Movies']
    )

    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Превышение среднего дохода ($)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
