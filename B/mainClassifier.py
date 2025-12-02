from sklearn.model_selection import train_test_split, KFold, cross_val_predict, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.base import clone
import numpy as np
import pandas as pd
import matplotlib as plt

SEED = 14
num_features = 4

#---------------------------------------------------------------------------------------------------
#EX1
#EX1.1 ##################################
#EX1.1.1
#Usar critério 70/30 
def create_split_train_test(X, y, test_size=0.30, random_state=SEED):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_test, y_train, y_test

#Usar critério 40/30/30
def create_split_tvt(X, y, val_size=0.30, test_size=0.30, random_state=SEED):
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
    return recall_score(y_true, y_pred, average=average, zero_division=0)

#EX1.2.3
def precision(y_true, y_pred, average='macro'):
    #Calcula a Precision.
    return precision_score(y_true, y_pred, average=average, zero_division=0)

#EX1.2.4
def f1(y_true, y_pred, average='macro'):
    #Calcula o F1-score.
    return f1_score(y_true, y_pred, average=average, zero_division=0)

# PRINTING AND SAVING METRICS
# def print_metrics(y_true, y_predict, label, printing=True):
#     metrics = {
#         "confusion_matrix": calcular_matriz_confusao(y_true, y_predict),
#         "recall": recall(y_true, y_predict),
#         "precision": precision(y_true, y_predict),
#         "f1-score": f1(y_true, y_predict)
#     }

#     if printing:
#         print(f"===== {label} =====")
#         print("Confusion Matrix:")
#         print(metrics["confusion_matrix"])
#         print()
#         print(f"Recall:          {metrics['recall']:.4f}")
#         print(f"Precision:       {metrics['precision']:.4f}")
#         print(f"F1-Score:        {metrics['f1-score']:.4f}")
#         print("=============================\n")

#     return metrics

# import numpy as np

def print_metrics(y_true, y_predict, label, printing=True):
    """
    Aceita:
      - y_true, y_predict (arrays individuais)
      - OU listas de y_true / y_predict vindos de K-folds.
    """

    # Detecta automaticamente se é K-fold (listas) ou caso único
    is_kfold = isinstance(y_true, list)

    if not is_kfold:
        # Caso normal: calcular tudo para uma única predição
        metrics = {
            "confusion_matrix": calcular_matriz_confusao(y_true, y_predict),
            "recall": recall(y_true, y_predict),
            "precision": precision(y_true, y_predict),
            "f1-score": f1(y_true, y_predict)
        }

    else:
        # Caso K-folds
        precisions = []
        recalls = []
        f1s = []
        confusion_matrices = []

        for yt, yp in zip(y_true, y_predict):
            confusion_matrices.append(calcular_matriz_confusao(yt, yp))
            recalls.append(recall(yt, yp))
            precisions.append(precision(yt, yp))
            f1s.append(f1(yt, yp))

        # guardar tudo no dicionário
        cms_mean, cms_std = mean_std_confusion_matrix(confusion_matrices)

        metrics = {
            "confusion_matrices_mean": cms_mean,
            "confusion_matrices_std": cms_std,
            "precision_mean": np.mean(precisions),
            "precision_std": np.std(precisions),
            "recall_mean": np.mean(recalls),
            "recall_std": np.std(recalls),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s)
        }

    if printing:
        print(f"\n===== {label} =====")

        if not is_kfold:
            print("Confusion Matrix:")
            print(metrics["confusion_matrix"])
            print()
            print(f"Recall:          {metrics['recall']:.4f}")
            print(f"Precision:       {metrics['precision']:.4f}")
            print(f"F1-Score:        {metrics['f1-score']:.4f}")

        else:
            print("K-Fold results (means ± std):\n")
            print(f"Precision:       {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
            print(f"Recall:          {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
            print(f"F1-Score:        {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
            print("\n(Confusion matrices individuais guardadas no dicionário.)")

        print("=============================\n")

    return metrics


#---------------------------------------------------------------------------------------------------
###########################################
################ EX2.1 ####################
###########################################

# --- Carregar dataset Iris ---
def load_data_set():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y

# Função para o classificador Random
def classifier_random(X_train, y_train, X_test, y_test, label ="", printing = True):
    clf = DummyClassifier(strategy='uniform', random_state=SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = print_metrics(y_test, y_pred, label, printing=printing)

    return metrics, y_pred

#Função para o classificador OneR
def classifier_oneR(X_train, y_train, X_test, y_test, max_depth=1, label = "", printing = True):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = print_metrics(y_test, y_pred, label, printing=printing)

    return metrics, y_pred

#Função para o classificador kNN
def classifier_kNN(X_train, y_train, X_test, y_test, k, label = "", printing = True):
    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metrics = print_metrics(y_test, y_pred, label, printing=printing)

    return metrics, y_pred

###########################################
################ MODELOS ##################
###########################################

def evaluate_with_kfold(X, y, classifier, rkf, label="KFOLD"):
    y_preds = []
    y_trues = []

    for train_idx, test_idx in rkf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone do classificador para não "acumular" treino entre folds
        clf = clone(classifier)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        y_preds.append(y_pred)
        y_trues.append(y_test)

    metrics = print_metrics(y_trues, y_preds, label=label)
    return metrics

def random_baseline_kfold(X, y, rkf):
    clf = DummyClassifier(strategy='uniform', random_state=SEED)
    return evaluate_with_kfold(X, y, clf, rkf, label="RANDOM Baseline - KFOLD")

def oneR_baseline_kfold(X, y, rkf):
    clf = DecisionTreeClassifier(max_depth=1, random_state=SEED)
    return evaluate_with_kfold(X, y, clf, rkf, label="ONE R Baseline - KFOLD")

def random_baseline(X, y):
    # TRAIN ONLY:
    metrics_to = classifier_random(X, y, X, y, label="RANDOM Baseline  - TRAIN ONLY")

    # Train, test sets:
    X_train_70, X_test_70, y_train_30, y_test_30 = create_split_train_test(X, y, test_size=0.3, random_state=SEED)
    metrics_tt = classifier_random(X_train_70, y_train_30, X_test_70, y_test_30, label="RANDOM Baseline - Train Test")

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
    metrics_cv = random_baseline_kfold(X, y, rkf)


def oneR_baseline(X, y):
        # TRAIN ONLY:
    metrics_to = classifier_oneR(X, y, X, y, label="ONE R TRAIN ONLY")

    # Train, test sets:
    X_train_70, X_test_70, y_train_30, y_test_30 = create_split_train_test(X, y, test_size=0.3, random_state=SEED)
    metrics_tt = classifier_oneR(X_train_70, y_train_30, X_test_70, y_test_30, label="ONE R Train Test")

    rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=SEED)
    metrics_cv = oneR_baseline_kfold(X, y, rkf)

def ex_2_1(X, y):

    # RANDOM BASELINE
    random_baseline(X, y)

    # ONER BASELIDE
    oneR_baseline(X, y)

    return

###########################################
################ EX2.2 ####################
###########################################

## EX2.2.1 ################################
def ex_2_2_1(X_iris, y_iris):

    k = 1
    clf_k = KNeighborsClassifier(n_neighbors = k)

    # TT
    clf_k.fit(X_iris, y_iris)
    y_pre_trainonly = clf_k.predict(X_iris)
    print_metrics(y_iris, y_pre_trainonly, "KNN FOR TRAIN ONLY")

    # TT 70-30
    X_train_70, X_test_30, y_train_70, y_test_30 = create_split_train_test(X_iris, y_iris, test_size=0.3, random_state=SEED)
    clf_k.fit(X_train_70, y_train_70)
    y_pre_tt_70_30 = clf_k.predict(X_test_30)
    print_metrics(y_test_30, y_pre_tt_70_30, "KNN FOR TRAIN TEST 70-30")

    # KFolds
    rkf = RepeatedStratifiedKFold(n_splits = 10, n_repeats=10, random_state=SEED)
    clf_for_kfolds = KNeighborsClassifier(n_neighbors=k)
    evaluate_with_kfold(X_iris, y_iris, clf_for_kfolds, rkf, label="KNN KFOLDS")

    # print_metrics(cms, precisions, recalls, f1s)


## EX2.2.2 ################################
# ⚠️ Pôr a guardar em CSV
def ex_2_to(X_iris, y_iris, min_range = 1, max_range = 15, step = 2):
    f1_scores = []

    for i in range(min_range, max_range + 1, step):
        k = i
        clf_k = KNeighborsClassifier(n_neighbors = k)

        clf_k.fit(X_iris, y_iris)
        y_pre_trainonly = clf_k.predict(X_iris)
        print_metrics(y_iris, y_pre_trainonly, f"Train only - {k}")
    
    return f1_scores

def ex_2_tvt(X_iris, y_iris, min_range = 1, max_range = 15, step = 2):

    f1_scores = []

    for i in range(min_range, max_range + 1, step):
        k = i
        clf_k = KNeighborsClassifier(n_neighbors = k)

        X_train_40, X_temp, y_train_40, y_temp = create_split_train_test(X_iris, y_iris, test_size=0.6, random_state=SEED)
        X_test_30, X_val_30, y_test_30, y_val_30 = create_split_train_test(X_temp, y_temp, test_size=0.5, random_state=SEED)
        clf_k.fit(X_train_40, y_train_40)
        y_pre_tt_40_30_30 = clf_k.predict(X_val_30)
        metrics = print_metrics(y_val_30, y_pre_tt_40_30_30, f"TT 40-30-30 - {k}")
        f1_scores.append(metrics['f1-score'])
    
    return f1_scores

def ex_2_cv(X_iris, y_iris, min_range = 1, max_range = 15, step = 2):
    
    f1_scores = []

    for i in range(min_range, max_range + 1, step):
        k = i
        rkf = RepeatedStratifiedKFold(n_splits = 10, n_repeats=10, random_state=SEED)
        clf_for_kfolds = KNeighborsClassifier(n_neighbors=k)
        evaluate_with_kfold(X_iris, y_iris, clf_for_kfolds, rkf, label="KNN KFOLDS")

    return

def ex_2_2_2(X_iris, y_iris, min_range = 1, max_range = 15, step = 2):
    print("=" * 40, "TO - Different k nearest")
    ex_2_to(X_iris, y_iris, min_range = min_range, max_range = max_range, step = step)

    print("=" * 40, "TVT - Different k nearest")
    ex_2_tvt(X_iris, y_iris, min_range = min_range, max_range = max_range, step = step)

    print("=" * 40, "10x10CV - Different k nearest")
    ex_2_cv(X_iris, y_iris, min_range = min_range, max_range = max_range, step = step)





###########################################
################ EX2.3 ####################
###########################################

def feature_ranking(X, y):
    
    f1_scores = []

    for i in range(num_features):
        f1 = ex_2_2_2(X[:, i], y[:, i], min_range = 3, max_range=3, step=2)
        f1_scores.append(f1)
    

    return

def ex_2_3():
    return

if __name__ == "__main__":
    print("Loading IRIS data...")
    X_iris, y_iris = load_data_set()
    print("Data loaded ✅")

    #EX2.1
    print("=" * 40, "EX 2.1 - BASELINE")
    ex_2_1(X_iris, y_iris)

    #EX2.2
    #EX2.2.1
    print("=" * 40, "EX 2.2.1 - TT, TVT, CV")
    ex_2_2_1(X_iris, y_iris)

    #EX2.2.2.
    print("=" * 40, "EX 2.2.2 - TT, TVT, CV - different k's nearest")
    ex_2_2_2(X_iris, y_iris)

    #EX2.3
