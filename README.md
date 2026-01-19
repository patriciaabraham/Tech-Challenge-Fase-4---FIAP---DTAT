# 📈 Preditor Ibovespa — Tech Challenge (FIAP)

Aplicação desenvolvida para o Tech Challenge da FIAP com o objetivo de disponibilizar um **modelo preditivo de séries temporais** em produção, por meio de uma aplicação interativa em **Streamlit**.

O modelo prevê se o **fechamento do índice Ibovespa do próximo dia útil será maior ou menor do que o fechamento do dia atual**.

---

## 🚀 Demonstração (Streamlit)

✅ A aplicação permite:

* Inserir dados do dia (ou usar dados históricos)
* Executar a previsão em tempo real
* Visualizar métricas do modelo
* Acompanhar gráficos e resultados da classificação

---

## 🧠 Sobre o modelo

O modelo utilizado foi uma **Regressão Logística** treinada com dados históricos do índice Ibovespa.

### 🎯 Objetivo do modelo

Prever a direção do mercado no próximo pregão:

* **1 → Alta** (fechamento de amanhã maior que o de hoje)
* **0 → Baixa** (fechamento de amanhã menor ou igual ao de hoje)

---

## 📊 Métricas de avaliação

As métricas podem ser consultadas diretamente na interface (sidebar do app), incluindo:

* Accuracy
* Precision
* Recall
* F1-Score
* Matriz de confusão

As métricas estão armazenadas em: `metrics.json`

---

## 🗂️ Estrutura do repositório

```bash
.
├── Dados Históricos - Ibovespa (5).csv   # Base histórica utilizada
├── ibovespa_app_vfa.py                  # Aplicação Streamlit
├── logreg.ipynb                         # Notebook do modelo e experimentos
├── logreg_pipeline.pkl                  # Pipeline do modelo treinado
├── metrics.json                         # Métricas do modelo
└── README.md                            # Este arquivo
```

---

## 🖥️ Como executar o projeto localmente

### 1) Clonar o repositório

```bash
git clone https://github.com/patriciaabraham/Tech-Challenge-Fase-4---FIAP---DTAT.git
cd Tech-Challenge-Fase-4---FIAP---DTAT
```

### 2) Criar e ativar um ambiente virtual (recomendado)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3) Instalar dependências

Se existir o arquivo requirements.txt:

```bash
pip install -r requirements.txt
```

Se não existir, instalar o mínimo:

```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

### 4) Rodar a aplicação Streamlit

```bash
streamlit run ibovespa_app_vfa.py
```

---

## 🔎 Observações importantes

* O arquivo `logreg_pipeline.pkl` deve estar na raiz do projeto para que o app carregue corretamente.
* Caso o repositório seja publicado em nuvem (Streamlit Community Cloud), é recomendável ter um `requirements.txt`.

---

## 👩‍💻 Autores

* Fillipe Júlio de Oliveira Nascimento
* Patrícia Vieira Abraham

---

## 📌 Tecnologias utilizadas

* Python
* Pandas / Numpy
* Scikit-learn
* Streamlit
* Plotly
* Joblib
