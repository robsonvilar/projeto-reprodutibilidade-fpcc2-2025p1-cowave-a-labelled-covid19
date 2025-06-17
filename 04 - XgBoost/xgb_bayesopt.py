# -*- coding: utf-8 -*-
"""
Created on Jun 4, 2022 7:44 AM
@author: melpakkampradeep
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer
import xgboost as xgb
import time

# Configurações de visualização do pandas
pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", None)
pd.set_option("expand_frame_repr", False)

# Leitura do dataset
datafull = pd.read_csv('COWAVE.csv')

# Separação de variáveis preditoras e alvo
X_t = datafull.drop(columns=['Wave', 'Date', 'Country_code'])
y_t = datafull[['Wave']]

# Separação treino/teste
X_train = X_t.iloc[0:149800].copy()
X_test = X_t.iloc[149801:-1].copy()
y_train = y_t.iloc[0:149800].copy()
y_test = y_t.iloc[149801:-1].copy()

# ======== Tratamento robusto de dados =========
# Remove colunas não numéricas
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Substitui inf/-inf por NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Remove valores muito altos (overflow)
limite = 1e10
X_train = X_train.applymap(lambda x: np.nan if abs(x) > limite else x)
X_test = X_test.applymap(lambda x: np.nan if abs(x) > limite else x)

# Remove colunas com muitos NaNs
X_train = X_train.dropna(axis=1)
X_test = X_test[X_train.columns]  # mantém mesmas colunas

# Preenche NaNs restantes com a média
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# ==============================================

# Modelo base
xgb_cl = xgb.XGBClassifier()
xgb_cl.fit(X_train, y_train)

# Avaliação inicial
preds = xgb_cl.predict(X_test)
predst = xgb_cl.predict(X_train)

print("Test acc: ", accuracy_score(y_test, preds))
print("Test rec:", recall_score(y_test, preds))
print("Test pres:", precision_score(y_test, preds))
print()
print("Train acc: ", accuracy_score(y_train, predst))
print("Train rec:", recall_score(y_train, predst))
print("Train pres:", precision_score(y_train, predst))

# Seleção de colunas para análise de importância
colunas_analise = ['T15','T16','T17','T18','T19','T20','T21']
X_train_imp = X_train[colunas_analise]
X_test_imp = X_test[colunas_analise]

# Avaliação de importância
model = XGBClassifier()
model.fit(X_train_imp, y_train)
importance = model.feature_importances_
for i, v in enumerate(importance):
    print(f'Feature: {colunas_analise[i]}, Score: {v:.5f}')

plt.figure(figsize=(30, 5))
plt.plot(colunas_analise, importance)
plt.show()

# Subconjunto final de features
colunas_final = ['MIN','Range','Sq', 'Median', 'Mean', 'Variance', 'MAX', 'PDF', 'Trend', 'Seasonal', 'Residual', 'T21', 'D7']
X_train = X_train[colunas_final].copy()
X_test = X_test[colunas_final].copy()

# Novo tratamento para segurança
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())

# Espaço de busca da otimização bayesiana
params_xgb = {
    'learning_rate': (0.0005, 1),
    'max_depth': (1, 10),
    'min_child_weight': (1, 10),
    'gamma': (0, 3),
    'colsample_bytree': (0.001, 1),
    'num_boost_round': (100, 500),
    'reg_lambda': (0.01, 10),
    'scale_pos_weight': (1, 10),
    'subsample': (0.001, 1),
}

def xgb_cl_bo(learning_rate, max_depth, min_child_weight, gamma, colsample_bytree, num_boost_round, reg_lambda, scale_pos_weight, subsample):
    params = {
        'learning_rate': learning_rate,
        'max_depth': round(max_depth),
        'min_child_weight': min_child_weight,
        'gamma': gamma,
        'colsample_bytree': colsample_bytree,
        'n_estimators': round(num_boost_round),
        'reg_lambda': reg_lambda,
        'scale_pos_weight': scale_pos_weight,
        'subsample': subsample
    }
    score = cross_val_score(XGBClassifier(random_state=123, **params), X_train, np.ravel(y_train), scoring=make_scorer(recall_score), cv=3).mean()
    return score

# Otimização bayesiana
start = time.time()
xgb_bo = BayesianOptimization(xgb_cl_bo, params_xgb, random_state=111)
xgb_bo.maximize(init_points=10, n_iter=5)  # Reduzido para testes rápidos
print('It takes %s minutes' % ((time.time() - start)/60))

# Resultados
params_xgb = xgb_bo.max['params']
params_xgb['max_depth'] = round(params_xgb['max_depth'])
params_xgb['n_estimators'] = round(params_xgb['num_boost_round'])
params_xgb.pop('num_boost_round')  # removido para evitar conflito
print("Melhores parâmetros encontrados:", params_xgb)

# Treinamento final com parâmetros otimizados
final_cl = xgb.XGBClassifier(**params_xgb)
final_cl.fit(X_train, y_train)

preds = final_cl.predict(X_test)
predst = final_cl.predict(X_train)

print("Test acc: ", accuracy_score(y_test, preds))
print("Test rec:", recall_score(y_test, preds))
print("Test pres:", precision_score(y_test, preds))
print()
print("Train acc: ", accuracy_score(y_train, predst))
print("Train rec:", recall_score(y_train, predst))
print("Train pres:", precision_score(y_train, predst))

# RandomizedSearch (como comparação)
param_grid = {
    'learning_rate': np.arange(0.05, 2, 0.05),
    'max_depth': np.arange(1, 10, 1),
    'min_child_weight': np.arange(1, 10, 0.5),
    'gamma': np.arange(0, 3, 0.1),
    'colsample_bytree': np.arange(0.1, 1, 0.05),
    'n_estimators': np.arange(100, 500, 50),
    'reg_lambda': np.arange(0.01, 10, 0.05),
    'scale_pos_weight': np.arange(1, 10, 0.5)
}

rand_cv = RandomizedSearchCV(xgb_cl, param_grid, n_iter=20, scoring="accuracy", n_jobs=-1, cv=3, verbose=True)
rand_cv.fit(X_train, y_train)

final_cl = xgb.XGBClassifier(**rand_cv.best_params_)
final_cl.fit(X_train, y_train)

preds = final_cl.predict(X_test)
predst = final_cl.predict(X_train)

print("Test acc: ", accuracy_score(y_test, preds))
print("Test rec:", recall_score(y_test, preds))
print("Test pres:", precision_score(y_test, preds))
print()
print("Train acc: ", accuracy_score(y_train, predst))
print("Train rec:", recall_score(y_train, predst))
print("Train pres:", precision_score(y_train, predst))
