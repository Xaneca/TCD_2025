from sklearn.model_selection import train_test_split, KFold, cross_val_predict, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

SEED = 14

#---------------------------------------------------------------------------------------------------
#EX1
#EX1.1 ##################################
#EX1.1.1
#Usar critério 70/30 
def create_split_train_test(X, y, test_size=0.30, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

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

def mean_std_confusion_matrix(list_of_cms):
    arr = np.stack(list_of_cms, axis=0)  # shape (n_folds, n_classes, n_classes)
    mean_cm = np.mean(arr, axis=0)
    std_cm  = np.std(arr, axis=0, ddof=0)
    return mean_cm, std_cm

#EX1.2.2
def recall(y_true, y_pred, average='macro'):
    #Calcula o Recall.
    #O parâmetro 'average' pode ser: 'binary', 'micro', 'macro', 'weighted'.
    return recall_score(y_true, y_pred, average=average)

#EX1.2.3
def precision(y_true, y_pred, average='macro'):
    #Calcula a Precision.
    return precision_score(y_true, y_pred, average=average)

#EX1.2.4
def f1(y_true, y_pred, average='macro'):
    #Calcula o F1-score.
    return f1_score(y_true, y_pred, average=average)

def print_metrics(y_true, y_predict, label, printing=True):
    mc = calcular_matriz_confusao(y_true, y_predict)
    re = recall(y_true, y_predict)
    pre = precision(y_true, y_predict)
    f1_s = f1(y_true, y_predict)

    if printing:
        print(label)
        print(mc)
        print(re)
        print(pre)
        print(f1_s)
    return mc, re, pre, f1_s

def print_metrics_kfolds(cms, precisions, recalls, f1s):

    mean_cm, std_cm = mean_std_confusion_matrix(cms)

    if print:
        print(">> 10x10-fold CV (100 folds) — k=1:")
        print("Precision (macro): mean={:.4f}, std={:.4f}".format(np.mean(precisions), np.std(precisions)))
        print("Recall (macro): mean={:.4f}, std={:.4f}".format(np.mean(recalls), np.std(recalls)))
        print("F1 (macro): mean={:.4f}, std={:.4f}".format(np.mean(f1s), np.std(f1s)))
        print("Matriz de Confusão (mean across folds):\n", np.round(mean_cm, 3))
        print("Matriz de Confusão (std across folds):\n", np.round(std_cm, 3))
    
    # fazer return dos valores
    return


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
    clf = DummyClassifier(strategy='uniform', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

#Função para o classificador OneR
def classifier_oneR(X_train, y_train, X_test, y_test, max_depth=1):
    clf = DecisionTreeClassifier(max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

#Função para o classificador kNN
def classifier_kNN(X_train, y_train, X_test, y_test, k):
    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

###########################################
################ EX2.2 ####################
## EX2.2.1 ################################
def ex_2_2_1(X_iris, y_iris):

    k = 1
    clf_k = KNeighborsClassifier(n_neighbors = k)

    # TT
    clf_k.fit(X_iris, y_iris)
    y_pre_trainonly = clf_k.predict(X_iris)
    print_metrics(y_iris, y_pre_trainonly, "METRICS FOR TRAIN ONLY")

    # TT 70-30

    X_train_70, X_test_30, y_train_70, y_test_30 = create_split_train_test(X_iris, y_iris, test_size=0.3, random_state=SEED)
    clf_k.fit(X_train_70, y_train_70)
    y_pre_tt_70_30 = clf_k.predict(X_test_30)
    print_metrics(y_test_30, y_pre_tt_70_30, "METRICS FOR TRAIN TEST 70-30")

    # KFolds
    rkf = RepeatedStratifiedKFold(n_splits = 10, n_repeats=10, random_state=SEED)
    cms = []
    precisions = []
    recalls = []
    f1s = []

    for train_indx, test_indx in rkf.split(X_iris, y_iris):
        X_tr, X_te = X_iris[train_indx], X_iris[test_indx]
        y_tr, y_te = y_iris[train_indx], y_iris[test_indx]
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        cms.append(calcular_matriz_confusao(y_te, y_pred))
        precisions.append(precision(y_te, y_pred))
        recalls.append(recall(y_te, y_pred))
        f1s.append(f1(y_te, y_pred))

    print_metrics_kfolds(cms, precisions, recalls, f1s)


# ⚠️ Pôr a guardar em CSV
def ex_2_2_2(X_iris, y_iris, min_range = 1, max_range = 15, step = 2):

    for i in range(min_range, max_range, step):
        k = i
        clf_k = KNeighborsClassifier(n_neighbors = k)

        clf_k.fit(X_iris, y_iris)
        y_pre_trainonly = clf_k.predict(X_iris)
        print_metrics(y_iris, y_pre_trainonly, f"T only - {k}", printing=False)

        X_train_40, X_temp, y_train_40, y_temp = create_split_train_test(X_iris, y_iris, train_size=0.4, random_state=SEED)
        X_test_30, X_val_30, y_test_30, y_val_30 = create_split_train_test(X_temp, y_temp, test_size=0.5, random_state=SEED)
        clf_k.fit(X_train_40, y_train_40)
        y_pre_tt_40_30_30 = clf_k.predict(X_test_30)
        print_metrics(y_test_30, y_pre_tt_40_30_30, f"TT 40-30-30 - {k}")




if __name__ == "__main__":

    #EX2.1

    #EX2.2
    X_iris, y_iris = load_data_set()

    #EX2.2.1
    ex_2_2_1(X_iris, y_iris)

    #EX2.2.2.
    ex_2_2_2(X_iris, y_iris)
