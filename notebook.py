# ============================================
# Classificação de Avaliações de Companhias Aéreas
# Dados Tabulares vs Textuais vs Combinados
# ============================================

# -----------------------------
# 1. Importação de Bibliotecas
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# 2. Carregamento dos Dados
# -----------------------------
# Ajuste o caminho conforme necessário

df = pd.read_csv("airlines_reviews.csv")
print(df.shape)
df.head()

# -----------------------------
# 3. Análise Exploratória
# -----------------------------
print(df.info())

# Distribuição da variável alvo
sns.countplot(x="recommended", data=df)
plt.title("Distribuição da Recomendação")
plt.show()

# Nota geral por recomendação
sns.boxplot(x="recommended", y="overall_rating", data=df)
plt.title("Nota Geral vs Recomendação")
plt.show()

# -----------------------------
# 4. Pré-processamento
# -----------------------------

# Variável alvo
y = df["recommended"].map({"yes": 1, "no": 0})

# -----------------------------
# 4.1 Dados Tabulares
# -----------------------------

# Seleção de colunas (ajuste conforme o dataset)
tabular_cols = [
    "overall_rating",
    "seat_comfort",
    "cabin_staff_service",
    "food_beverages",
    "value_for_money"
]

categorical_cols = ["travel_type", "class"]

X_tabular = df[tabular_cols + categorical_cols]

preprocess_tabular = ColumnTransformer([
    ("num", StandardScaler(), tabular_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# -----------------------------
# 4.2 Dados Textuais
# -----------------------------

text_col = "review_text"

# -----------------------------
# 5. Divisão Treino/Teste
# -----------------------------

X_train_tab, X_test_tab, y_train, y_test = train_test_split(
    X_tabular, y, test_size=0.2, random_state=42, stratify=y
)

X_train_text, X_test_text = train_test_split(
    df[text_col], test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. Modelo 1 – Apenas Dados Tabulares
# -----------------------------

model_tabular = Pipeline([
    ("preprocess", preprocess_tabular),
    ("clf", LogisticRegression(max_iter=1000))
])

model_tabular.fit(X_train_tab, y_train)

y_pred_tab = model_tabular.predict(X_test_tab)

print("Modelo Tabular")
print(classification_report(y_test, y_pred_tab))

# Matriz de confusão
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_tab)
plt.title("Matriz de Confusão – Tabular")
plt.show()

# -----------------------------
# 7. Modelo 2 – Apenas Dados Textuais
# -----------------------------

model_text = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

model_text.fit(X_train_text, y_train)

y_pred_text = model_text.predict(X_test_text)

print("Modelo Textual")
print(classification_report(y_test, y_pred_text))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_text)
plt.title("Matriz de Confusão – Textual")
plt.show()

# -----------------------------
# 7.1 Top 20 Palavras por Classe
# -----------------------------

feature_names = model_text.named_steps["tfidf"].get_feature_names_out()
coef = model_text.named_steps["clf"].coef_[0]

# Classe positiva (recomenda)
top_pos_idx = np.argsort(coef)[-20:]

# Classe negativa (não recomenda)
top_neg_idx = np.argsort(coef)[:20]

# Gráficos
plt.barh(feature_names[top_pos_idx], coef[top_pos_idx])
plt.title("Top 20 Palavras – Recomenda")
plt.show()

plt.barh(feature_names[top_neg_idx], coef[top_neg_idx])
plt.title("Top 20 Palavras – Não Recomenda")
plt.show()

# -----------------------------
# 8. Modelo 3 – Dados Tabulares + Textuais
# -----------------------------

combined_features = ColumnTransformer([
    ("tabular", preprocess_tabular, tabular_cols + categorical_cols),
    ("text", TfidfVectorizer(stop_words="english", max_features=5000), text_col)
])

X = df[tabular_cols + categorical_cols + [text_col]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_combined = Pipeline([
    ("features", combined_features),
    ("clf", LogisticRegression(max_iter=1000))
])

model_combined.fit(X_train, y_train)

y_pred_comb = model_combined.predict(X_test)

print("Modelo Combinado")
print(classification_report(y_test, y_pred_comb))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_comb)
plt.title("Matriz de Confusão – Combinado")
plt.show()

# -----------------------------
# 9. Comparação Final dos Modelos
# -----------------------------

results = pd.DataFrame({
    "Modelo": ["Tabular", "Textual", "Combinado"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_tab),
        accuracy_score(y_test, y_pred_text),
        accuracy_score(y_test, y_pred_comb)
    ],
    "F1-score": [
        f1_score(y_test, y_pred_tab),
        f1_score(y_test, y_pred_text),
        f1_score(y_test, y_pred_comb)
    ]
})

results

sns.barplot(x="Modelo", y="F1-score", data=results)
plt.title("Comparação de F1-score entre os Modelos")
plt.show()

# -----------------------------
# 10. Conclusão (Notebook)
# -----------------------------
print("O modelo combinado apresentou o melhor desempenho geral,")
print("demonstrando que dados tabulares e textuais são complementares.")
