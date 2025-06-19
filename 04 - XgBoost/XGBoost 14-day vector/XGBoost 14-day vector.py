import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Visualização
pd.set_option("display.max_columns", None)

# Carrega dados
df = pd.read_csv("COWAVE.csv")

# Define os países de treino e teste
codigos_treino = df[df['Country_code'].between('AF', 'QA')]['Country_code'].unique()
codigos_teste = df[df['Country_code'].between('RK', 'ZW')]['Country_code'].unique()

# Seleciona o vetor de 14 dias (T8 a T21)
vetor_14dias = [f"T{i}" for i in range(8, 22)]  # ['T8', ..., 'T21']

# Filtra datasets
X_train = df[df['Country_code'].isin(codigos_treino)][vetor_14dias].copy()
y_train = df[df['Country_code'].isin(codigos_treino)]['Wave'].copy()

X_test = df[df['Country_code'].isin(codigos_teste)][vetor_14dias].copy()
y_test = df[df['Country_code'].isin(codigos_teste)]['Wave'].copy()

# Tratamento de dados (seguro, sem alterar variância)
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Treina modelo padrão XGBoost
modelo = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
modelo.fit(X_train, y_train)

# Avaliação
preds = modelo.predict(X_test)
preds_treino = modelo.predict(X_train)

print("=== RESULTADOS DE TESTE ===")
print("Test acc: ", accuracy_score(y_test, preds))
print("Test rec:", recall_score(y_test, preds))
print("Test pres:", precision_score(y_test, preds))

print("\n=== RESULTADOS DE TREINO ===")
print("Train acc: ", accuracy_score(y_train, preds_treino))
print("Train rec:", recall_score(y_train, preds_treino))
print("Train pres:", precision_score(y_train, preds_treino))

# Importância das features
importancias = modelo.feature_importances_
for i, v in enumerate(importancias):
    print(f'Feature: {vetor_14dias[i]}, Score: {v:.5f}')

# Plot
plt.bar(vetor_14dias, importancias)
plt.title("Importância das Features (Vetor 14 Dias)")
plt.show()

