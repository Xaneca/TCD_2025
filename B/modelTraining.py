from sklearn.model_selection import train_test_split, KFold, cross_val_predict, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.base import clone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skrebate import ReliefF
import joblib

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import product
import random
import ast

SEED = 42

def split_set(X, y, train_size = 0.4, val_size = 0.3, test_size = 0.3, random_state=38):
    if val_size > 0:
        # Primeiro separa o conjunto de teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Proporção relativa da validação dentro de X_temp (treino + validação)
        val_relative_size = val_size / (1.0 - test_size)

        # Depois separa treino e validação, estratificando por y_temp (não por y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=random_state, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

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

def print_metrics(y_true, y_pred, label="Metric Results", printing = True):
    """
    y_true e y_pred podem ser:
      - arrays únicos
      - listas de arrays (K-Fold)
    """
    is_kfold = isinstance(y_true, list)

    if not is_kfold:
        cm = calcular_matriz_confusao(y_true, y_pred)
        rec = recall(y_true, y_pred)
        prec = precision(y_true, y_pred)
        f1s = f1(y_true, y_pred)

        if printing:
            print(f"\n===== {label} =====")
            print("Confusion Matrix:")
            print(cm)
            print(f"Recall:    {rec:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"F1-Score:  {f1s:.4f}")
            print("=========================")

        return {"confusion_matrix": cm, "recall": rec, "precision": prec, "f1-score": f1s}

    else:
        recalls, precisions, f1s, cms = [], [], [], []
        for yt, yp in zip(y_true, y_pred):
            recalls.append(recall(yt, yp))
            precisions.append(precision(yt, yp))
            f1s.append(f1(yt, yp))
            cms.append(calcular_matriz_confusao(yt, yp))

        if printing:
            print(f"\n===== {label} - K-Fold results =====")
            print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
            print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
            print(f"F1-Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print("(Confusion matrices individuais guardadas no array.)")
            print("=========================")

        return {"confusion_matrices": cms, 
                "recall_mean": np.mean(recalls), "recall_std": np.std(recalls),
                "precision_mean": np.mean(precisions), "precision_std": np.std(precisions),
                "f1_mean": np.mean(f1s), "f1_std": np.std(f1s)}

def classifier_model(model, X_train, y_train, X_test, y_test, label = "", printing=True):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metrics_df = metrics_to_dataframe(y_test, y_pred, label)
    metrics = print_metrics(y_test, y_pred, label=label, printing=printing)

    return metrics

def reset_model(model):
    """
    Cria uma nova instância do modelo com os mesmos parâmetros base.
    Garante que o modelo vem "limpo" e sem treino.
    """
    return model.__class__(**model.get_params())


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Modelo salvo em {filename}")

##############################################################################

def compute_feature_ranking(X, y):
    relief = ReliefF(n_neighbors=20)
    relief.fit(X, y)
    scores = relief.feature_importances_

    return np.argsort(scores)[::-1]     # ordem decrescente

def metrics_summary_features(metrics_list):
    n_features = [m['n_features'] for m in metrics_list]
    recall = []
    precision = []
    f1 = []

    for m in metrics_list:
        recall.append(m['recall'])
        precision.append(m['precision'])
        f1.append(m['f1-score'])

    return pd.DataFrame({
        'n_features': n_features,
        'recall': recall,
        'precision': precision,
        'f1': f1
    })

def featureRanking(X_train, y_train, X_test, y_test, model, plot=True, printing=True):
    scores = compute_feature_ranking(X_train, y_train)
    if printing:
        print("Score:")
        print(scores)

    metrics_list = []

    for i in range(len(scores)):
        X_train_i = X_train[:, scores[:i+1]]
        X_test_i = X_test[:, scores[:i+1]]

        clf = reset_model(model)

        metrics = classifier_model(clf, X_train_i, y_train, X_test_i, y_test, "", False)

        metrics['n_features'] = i+1
        metrics_list.append(metrics)

    df_feat = metrics_summary_features(metrics_list)
    # display(df_feat)

    best_idx = df_feat['f1'].idxmax()
    best_n = df_feat.loc[best_idx, 'n_features']
    best_features = scores[:best_n]

    if printing:
        print(f"TOP {len(best_features)} features")
        print("BEST FEATURES", best_features)

    if plot:
        # Plot do F1
        plt.figure(figsize=(6,4))
        plt.plot(df_feat['n_features'], df_feat['f1'], marker='o')
        plt.xlabel("Número de features")
        plt.ylabel("F1-Score")
        plt.title("F1-Score por número de features selecionadas")
        plt.grid(True)
        plt.show()
    
    return best_features


def chooseParameters(X_train, y_train, X_test, y_test, model, bfs, param_grid):
    """
    Itera sobre todas as combinações de parâmetros e devolve o melhor com F1-score.
    """
    keys = list(param_grid.keys())
    valores = list(param_grid.values())
    best_score = -1
    best_params = None

    # Parâmetros base (sem treino)
    base_params = model.get_params()

    # Itera todas as combinações
    for comb in product(*valores):
        params = dict(zip(keys, comb))

        # Criar novo modelo "limpo"
        clf = reset_model(model)

        clf.set_params(**params)
        
        clf.fit(X_train[:, bfs], y_train)     # com best feature set
        y_pred = clf.predict(X_test[:, bfs])
        score = f1_score(y_test, y_pred, average='macro')  # macro ou weighted

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

def deployModel(X_train_orig, y_train_orig, X_test, y_test, model, bfs, parameters, filename, label = "",):
    
    clf = reset_model(model)

    # model, X_train, y_train, X_test, y_test, label = "", printing=True
    metrics = classifier_model(clf, X_train_orig[:, bfs], y_train_orig, X_test[:,bfs], y_test, label=label, printing=True)

    clf = reset_model(model)

    X_train = np.vstack((X_train_orig, X_test))         # vertical stack
    y_train = np.concatenate((y_train_orig, y_test))    # concatenate because 1 dimension

    clf.fit(X_train, y_train)

    save_model(clf, filename)

    return metrics

def createFolds(X, y, n_folds=10, n_repeats=10):
    folds = []

    rkf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=SEED)

    for train_idx, test_idx in rkf.split(X, y):
        # train (90%) + test (10%)
        X_train_orig, X_test = X[train_idx], X[test_idx]
        y_train_orig, y_test = y[train_idx], y[test_idx]

        # train (90% * 0.9) + validation (90% * 0.1)
        # 80 + 10 + 10
        X_train, X_val, y_train, y_val = split_set(X_train_orig, y_train_orig, train_size = 0.9, val_size = 0, test_size = 0.1, random_state=SEED)

        dic = {"X_train_orig": X_train_orig, 
            "y_train_orig": y_train_orig,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_val": X_val,
            "y_val": y_val}
        
        folds.append(dic)
        
    return folds

def train_tvt(X, y, model, parameters, random_state, filename, label):
    dic = split_set(X, y, random_state=random_state)
    X_train = dic["X_train"]
    X_val = dic["X_val"]
    y_train = dic["y_train"]
    y_val = dic["y_val"]

    bfs = featureRanking(X_train, y_train, X_val, y_val, model)
    best_parameters = chooseParameters(X_train, y_train, X_val, y_val, model, bfs, parameters)
    metrics = deployModel(dic["X_train_orig"], dic["y_train_orig"], dic["X_test"], dic["y_test"], model, bfs, parameters, filename, label=label)
    return metrics

def train_cv(X, y, model, parameters, filename, random_state = SEED, n_folds = 10, n_repeats = 10):
    dic = split_set()

    return

def run_model(X, y, model, split_scheme, parameters, filename, label="", random_state=SEED):
    if split_scheme == "TVT":
        return train_tvt(X, y, model, parameters, random_state, filename)
    elif split_scheme == "CV":
        if len(model) != len(parameters):
            print("És burro")
        train_cv(X, y, model, parameters, filename, random_state=random_state)

    return