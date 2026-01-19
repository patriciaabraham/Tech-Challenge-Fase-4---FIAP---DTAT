import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime
import os
import numpy as np
from pathlib import Path

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsão de Tendência do Ibovespa - Tech Challenge", layout="wide")

# --- PATH BASE (evita erro de caminho ao rodar no VS Code/Streamlit) ---
BASE_DIR = Path(__file__).resolve().parent

# --- CARREGAMENTO DE DADOS E MODELO ---
@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "logreg_pipeline.pkl")

@st.cache_data
def load_metrics():
    with open(BASE_DIR / "metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_historical_data(file_path="Dados Históricos - Ibovespa (5).csv"):
    file_path = BASE_DIR / file_path
    if file_path.exists():
        df = pd.read_csv(file_path)

        # Ajuste: garantir que existe a coluna Data e está em datetime
        if "Data" not in df.columns:
            st.error("O CSV histórico precisa ter a coluna 'Data'.")
            return pd.DataFrame()

        df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y", errors="coerce")
        df = df.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)
        return df
    else:
        st.error(f"Arquivo histórico '{file_path}' não encontrado.")
        return pd.DataFrame()

# --- ENGENHARIA DE FEATURES ---
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Garante colunas necessárias
    required_cols = ["Último", "Máxima", "Mínima", "Abertura"]
    for c in required_cols:
        if c not in df.columns:
            # cria coluna com 0 se faltar (evita KeyError)
            df[c] = 0.0

    # Variação percentual (robusta)
    if "Var%" not in df.columns:
        df["Var%"] = 0.0
    else:
        # Se já vier numérico, converte de boa; se vier string, trata
        df["Var%"] = (df["Var%"].astype(str)
                      .str.replace(",", ".", regex=False)
                      .str.replace("%", "", regex=False))
        df["Var%"] = pd.to_numeric(df["Var%"], errors="coerce").fillna(0.0) / 100

    df["pct_change_close"] = df["Último"].pct_change() * 100
    df["diff_close"] = df["Último"].diff()
    df["high_low_diff"] = df["Máxima"] - df["Mínima"]
    df["open_close_diff"] = df["Abertura"] - df["Último"]

    # Simple Moving Averages (SMA)
    for window in [3, 7, 14, 21, 30]:
        df[f"SMA_{window}"] = df["Último"].rolling(window=window).mean()
        df[f"SMA_Vol_{window}"] = df["Var%"].rolling(window=window).mean()

    # Volatility (Standard Deviation)
    for window in [7, 21]:
        df[f"Vol_Std_{window}"] = df["Último"].rolling(window=window).std()

    df = df.fillna(0)
    return df

# --- INICIALIZA ---
model = load_model()
metrics = load_metrics()
historical_data = load_historical_data()

# Features esperadas pelo modelo (para evitar mismatch)
expected_features = list(getattr(model, "feature_names_in_", []))

# --- HEADER ---
st.title("🚀 Ibovespa Trend Predictor")
st.markdown("""
Esta aplicação utiliza um modelo de *Regressão Logística* para prever se o fechamento do Ibovespa de amanhã será *maior* que o de hoje.
""")

# --- SIDEBAR: MÉTRICAS DE VALIDAÇÃO ---
st.sidebar.header("📊 Performance do Modelo")
st.sidebar.metric("Acurácia (Teste)", f"{metrics.get('accuracy_test', 0):.2%}")
st.sidebar.metric("F1-Score", f"{metrics.get('f1_test', 0):.2f}")

with st.sidebar.expander("Ver Relatório Completo"):
    st.text(metrics.get("classification_report", ""))

# --- ÁREA PRINCIPAL: INPUT DE DADOS ---
st.subheader("📝 Previsão de Nova Tendência")

st.markdown("Insira os valores mais recentes do Ibovespa para gerar uma previsão:")
col1, col2, col3 = st.columns(3)

with col1:
    current_close = st.number_input("Fechamento de Hoje", value=117800.0, format="%.2f")
    current_high = st.number_input("Máxima de Hoje", value=118000.0, format="%.2f")
    current_low = st.number_input("Mínima de Hoje", value=117000.0, format="%.2f")

with col2:
    current_open = st.number_input("Abertura de Hoje", value=117500.0, format="%.2f")
    current_volume = st.number_input("Volume de Hoje", value=23500000.0, format="%.0f")

# Botão para gerar a previsão
if st.button("Executar Previsão"):
    if not historical_data.empty:
        temp_df = historical_data.copy()

        # Adiciona a linha mais recente (PADRONIZADO com o CSV: Data/Abertura/Máxima/Mínima/Último/Var%)
        new_row = pd.DataFrame([{
            "Data": pd.to_datetime(datetime.now().date()),
            "Abertura": float(current_open),
            "Máxima": float(current_high),
            "Mínima": float(current_low),
            "Último": float(current_close),
            "Var%": 0.0,  # sem ATR/Parkinson; aqui mantemos simples
        }])

        temp_df = pd.concat([temp_df, new_row], ignore_index=True)
        temp_df = temp_df.sort_values("Data").reset_index(drop=True)

        # Recria as features
        df_with_features = create_features(temp_df)

        latest_row = df_with_features.tail(1).copy()

        # (Simples) Features de data, somente se o modelo pedir
        if "Data" in latest_row.columns and expected_features:
            if "day_of_week" in expected_features and "day_of_week" not in latest_row.columns:
                latest_row["day_of_week"] = latest_row["Data"].dt.dayofweek
            if "day_of_month" in expected_features and "day_of_month" not in latest_row.columns:
                latest_row["day_of_month"] = latest_row["Data"].dt.day
            if "month" in expected_features and "month" not in latest_row.columns:
                latest_row["month"] = latest_row["Data"].dt.month
            if "year" in expected_features and "year" not in latest_row.columns:
                latest_row["year"] = latest_row["Data"].dt.year

        # Monta input EXATAMENTE com as colunas do modelo (sem mexer no modelo)
        if expected_features:
            missing = [c for c in expected_features if c not in latest_row.columns]
            input_for_prediction = latest_row.reindex(columns=expected_features, fill_value=0)

            # opcional: avisar o que foi preenchido com 0
            if missing:
                st.caption(
                    "Features faltando preenchidas com 0: "
                    + ", ".join(missing[:20])
                    + (" ..." if len(missing) > 20 else "")
                )
        else:
            # fallback (caso o modelo não exponha feature_names_in_)
            st.warning("O modelo não expõe feature_names_in_. Tentando prever com as colunas disponíveis.")
            input_for_prediction = latest_row.drop(columns=["Data"], errors="ignore")

        # Previsão
        prediction = model.predict(input_for_prediction)[0]
        prob = model.predict_proba(input_for_prediction)[0][1]

        # Exibição do Resultado
        if prediction == 1:
            st.success(f"### 📈 Tendência de ALTA ({prob:.2%})")
        else:
            st.error(f"### 📉 Tendência de QUEDA ({(1 - prob):.2%})")

        # Gráfico de "Velocímetro" de Probabilidade
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Confiança da Alta (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 100], "color": "lightgreen"},
                ],
            },
        ))
        st.plotly_chart(fig, use_container_width=True)

        # --- SISTEMA DE LOGS ---
        log_entry = input_for_prediction.iloc[0].to_dict()
        log_entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry["prediction"] = int(prediction)
        log_entry["probability"] = float(prob)

        log_file = BASE_DIR / "usage_log.csv"
        log_df = pd.DataFrame([log_entry])

        if not log_file.exists():
            log_df.to_csv(log_file, index=False)
        else:
            log_df.to_csv(log_file, mode="a", header=False, index=False)

        st.info(f"Log de uso atualizado em {log_file.name}")
    else:
        st.warning("Não foi possível gerar a previsão sem dados históricos.")

# --- GRÁFICO DE MÉTRICAS DA SÉRIE TEMPORAL ---
st.divider()
st.subheader("📈 Análise de Tendências Históricas")
st.markdown("Explore como as médias móveis e a volatilidade se comportam ao longo do tempo.")

if not historical_data.empty:
    df_with_features_history = create_features(historical_data.copy())

    metric_options = [
        "Último", "SMA_3", "SMA_7", "SMA_14", "SMA_21", "SMA_30",
        "Vol_Std_7", "Vol_Std_21", "SMA_Vol_30", "pct_change_close"
    ]

    selected_metrics = st.multiselect(
        "Selecione as métricas para exibir no gráfico:",
        options=metric_options,
        default=["Último", "SMA_7", "SMA_30"],
    )

    if selected_metrics:
        fig_series = go.Figure()
        for metric in selected_metrics:
            if metric in df_with_features_history.columns:
                fig_series.add_trace(go.Scatter(
                    x=df_with_features_history["Data"],
                    y=df_with_features_history[metric],
                    mode="lines",
                    name=metric,
                ))

        fig_series.update_layout(
            title="Evolução das Métricas do Ibovespa",
            xaxis_title="Data",
            yaxis_title="Valor",
            hovermode="x unified",
            height=500,
        )
        st.plotly_chart(fig_series, use_container_width=True)
    else:
        st.info("Selecione pelo menos uma métrica para visualizar o gráfico histórico.")
else:
    st.info("Carregue os dados históricos para ver a análise de tendências.")

