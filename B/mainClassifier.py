from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

#---------------------------------------------------------------------------------------------------
#EX1
#EX1.1 ##################################
#EX1.1.1
#Usar critério 70/30 
def create_split_train_test(X, y, test_size=0.30, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    return X_train, X_test, y_train, y_test

#Usar critério 40/30/30
def create_split_tvt(X, y, val_size=0.30, test_size=0.30, random_state=42):
    # Primeiro, separa o conjunto de teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Calcula a proporção da validação em relação ao restante (treino + validação)
    val_relative_size = val_size / (1 - test_size)

    # Depois, separa treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

#EX1.1.2
def create_split_kfold(X, y, n_splits=5, random_state=None, shuffle=False):

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    folds = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((fold, X_train, X_test, y_train, y_test))

    return folds

#EX 1.2 ##################################
#EX 1.2.1
def calcular_matriz_confusao(y_true, y_pred):
    #Retorna a matriz de confusão.
    return confusion_matrix(y_true, y_pred)

#EX1.2.2
def recall_macro(y_true, y_pred, average='macro'):
    #Calcula o Recall.
    #O parâmetro 'average' pode ser: 'binary', 'micro', 'macro', 'weighted'.
    return recall_score(y_true, y_pred, average=average)

#EX1.2.3
def precision_macro(y_true, y_pred, average='macro'):
    #Calcula a Precision.
    return precision_score(y_true, y_pred, average=average)

#EX1.2.4
def f1_macro(y_true, y_pred, average='macro'):
    #Calcula o F1-score.
    return f1_score(y_true, y_pred, average=average)

#---------------------------------------------------------------------------------------------------
#EX2
#EX2.1 ##################################
# --- Carregar dataset Iris ---
def load_data_set():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

# Função para o classificador Random
def classifier_random(X_train, y_train, X_test, y_test):
    DummyClassifier(strategy='uniform', random_state=42).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

#Função para o classificador OneR
def classifier_oneR(X_train, y_Train, X_test, y_test, max_depth=1):
    DecisionTreeClassifier(max_depth, random_state=42).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

#Função para o classificador kNN
def classifier_kNN(X_train, y_train, X_test, y_test, k):
    KNeighborsClassifier(k).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred






if __name__ == "__main__":

    #EX2.2
