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
from sklearn.preprocessing import MinMaxScaler          # Normalizaçao
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from itertools import product
import random
import ast

SEED = 42

import numpy as np

def split_dataset_by_person_tvt(X, y, person_col_index=-1):
    """
    Divide o dataset em treino, validação e teste com base nas pessoas.

    X : np.ndarray
        Features, incluindo a coluna 'person'
    y : np.ndarray
        Labels (atividade)
    person_col_index : int
        Índice da coluna 'person' em X (default: última)

    Retorna
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """

    # Extrair coluna 'person'
    persons = X[:, person_col_index]

    # Definir grupos de pessoas
    train_persons = np.arange(1, 8)    # pessoas 1–7
    val_persons   = np.arange(8, 12)   # pessoas 8–11
    test_persons  = np.arange(12, 16)  # pessoas 12–15

    # Criar máscaras
    train_mask = np.isin(persons, train_persons)
    val_mask   = np.isin(persons, val_persons)
    test_mask  = np.isin(persons, test_persons)

    # Remover coluna 'person' de X
    X_features = np.delete(X, person_col_index, axis=1)

    # Dividir
    X_train = X_features[train_mask]
    y_train = y[train_mask]

    X_val = X_features[val_mask]
    y_val = y[val_mask]

    X_test = X_features[test_mask]
    y_test = y[test_mask]

    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Validação: {X_val.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")

    return X_train, y_train, X_val, y_val, X_test, y_test

import numpy as np

def split_dataset_by_person_tt(X, y, person_col_index=-1):
    """
    Divide o dataset em treino e teste com base nas pessoas.

    X : np.ndarray
        Features, incluindo a coluna 'person'
    y : np.ndarray
        Labels (atividade)

    Retorna
    -------
    X_train, y_train, X_test, y_test
    """

    # Extrair coluna 'person'
    persons = X[:, person_col_index]

    # Definir grupos de pessoas
    train_persons = np.arange(1, 12)   # pessoas 1–11
    test_persons  = np.arange(12, 16)  # pessoas 12–15

    # Criar máscaras
    train_mask = np.isin(persons, train_persons)
    test_mask  = np.isin(persons, test_persons)

    # Remover coluna 'person' das features
    X_features = np.delete(X, person_col_index, axis=1)

    # Dividir
    X_train = X_features[train_mask]
    y_train = y[train_mask]

    X_test = X_features[test_mask]
    y_test = y[test_mask]

    print(f"Treino: {X_train.shape[0]} amostras")
    print(f"Teste: {X_test.shape[0]} amostras")

    return X_train, y_train, X_test, y_test

def split_set(X, y, train_size = 0.4, val_size = 0.3, test_size = 0.3, random_state=38, use_iris=True):
    if (use_iris==True):
        if val_size > 0:
            # Primeiro separa o conjunto de teste
            X_train_orig, X_test, y_train_orig, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            # Proporção relativa da validação dentro de X_temp (treino + validação)
            val_relative_size = val_size / (1.0 - test_size)

            # Depois separa treino e validação, estratificando por y_temp (não por y)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_orig, y_train_orig, test_size=val_relative_size, random_state=random_state, stratify=y_train_orig
            )

            return {"X_train_orig": X_train_orig, "X_train": X_train, "X_test": X_test, "X_val": X_val, "y_train_orig": y_train_orig, "y_train": y_train, "y_test": y_test, "y_val": y_val}
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            return X_train, X_test, y_train, y_test
    else:
        if val_size > 0:
            X_train, y_train, X_val, y_val, X_test, y_test = split_dataset_by_person_tvt(X, y)

            # NORMALIZAÇÃO
            scaler = MinMaxScaler()

            # Normaliza para os máximos e minimos do X_Train e guarda
            X_train_scaled = scaler.fit_transform(X_train)

            # Normaliza com os máximos e minimos guardados
            X_val_scaled = scaler.transform(X_val)

            X_test_scaled = scaler.transform(X_test)

            X_train_orign = np.vstack((X_train_scaled, X_val_scaled))

            y_train_orign = np.concatenate((y_train, y_val), axis=0)

            return {"X_train_orig": X_train_orign, "X_train": X_train_scaled, "X_test": X_test_scaled, "X_val": X_val_scaled, "y_train_orig": y_train_orign, "y_train": y_train, "y_test": y_test, "y_val": y_val}
        else:
            X_train, y_train, X_test, y_test = split_dataset_by_person_tt(X, y)

            # NORMALIZAÇÃO
            scaler = MinMaxScaler()

            # Normaliza para os máximos e minimos do X_Train e guarda
            X_train_scaled = scaler.fit_transform(X_train)

            # Normaliza com os máximos e minimos guardados
            X_test_scaled = scaler.transform(X_test)

            return X_train_scaled, X_test_scaled, y_train, y_test

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

def reset_model(model):
    """
    Cria uma nova instância do modelo com os mesmos parâmetros base.
    Garante que o modelo vem "limpo" e sem treino.
    """
    return model.__class__(**model.get_params())


def classifier_model(model, X_train, y_train, X_test, y_test, label = "", printing=True, params=None):
    clf = reset_model(model)

    if params is not None: 
        clf.set_params(**params)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # metrics_df = metrics_to_dataframe(y_test, y_pred, label)
    metrics = print_metrics(y_test, y_pred, label=label, printing=printing)

    return metrics

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Modelo salvo em {filename}")

##############################################################################

def compute_feature_ranking(X, y, printing=True):
    relief = ReliefF(n_neighbors=20)
    relief.fit(X, y)
    scores = relief.feature_importances_

    scores = np.argsort(scores)[::-1]

    if printing:
        print("Score:")
        print(scores)

    return scores     # ordem decrescente

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

def pick_first_param_values(param_grid):
    """
    Recebe um dicionário de listas e devolve um dicionário onde
    cada chave fica com **apenas o primeiro valor**, sem lista.
    """
    new_grid = {}
    for key, values in param_grid.items():
        if isinstance(values, list) and len(values) > 0:
            new_grid[key] = values[0]  # **apenas o valor**
        else:
            new_grid[key] = values
    return new_grid

def featureRanking(X_train, y_train, X_test, y_test, model, scores, params=None, plot=True, printing=True, save=True, filename=None, title=None):

    metrics_list = []

    for i in range(len(scores)):
        X_train_i = X_train[:, scores[:i+1]]
        X_test_i = X_test[:, scores[:i+1]]

        clf = reset_model(model)

        metrics = classifier_model(clf, X_train_i, y_train, X_test_i, y_test, params=params, label=f"features: {scores[:i+1]}", printing=False)

        metrics['n_features'] = i+1
        metrics_list.append(metrics)

    df_feat = metrics_summary_features(metrics_list)
    if printing:
        display(df_feat)

    best_idx = df_feat['f1'].idxmax()
    best_n = df_feat.loc[best_idx, 'n_features']
    best_features = scores[:best_n]

    if printing:
        print(f"TOP {len(best_features)} features")
        print("BEST FEATURES", best_features)

    # Plot do F1
    plt.figure(figsize=(6,4))
    plt.plot(df_feat['n_features'], df_feat['f1'], marker='o')
    plt.xlabel("Número de features")
    plt.ylabel("F1-Score")
    plt.title(f"F1-Score/nº features {title}")
    plt.grid(True)
    if save:
        plt.savefig(filename)
    if plot:
        plt.show()
    plt.close()
    
    return best_features, metrics_list


def chooseParameters(X_train, y_train, X_test, y_test, model, bfs, param_grid):
    """
    Itera sobre todas as combinações de parâmetros e devolve o melhor com F1-score.
    """
    keys = list(param_grid.keys())
    valores = list(param_grid.values())
    best_score = -1
    best_params = None

    params_list = []

    # ver nº total de combinaçoes
    total_combs = 1
    for v in valores:
        total_combs *= len(v)

    if total_combs > 1: 
        # Itera todas as combinações
        for comb in product(*valores):
            params = dict(zip(keys, comb))

            metrics = classifier_model(
                model,
                X_train[:, bfs], 
                y_train,
                X_test[:, bfs],
                y_test,
                printing=False,
                params=params
            )

            score = metrics["f1-score"]   # AJUSTAR SE NECESSÁRIO
            params_list.append([params, metrics])

            if score > best_score:
                best_score = score
                best_params = params

        print("Best Parameters:", best_params)

        return best_params, best_score, params_list
    else:
        params = {k: v[0] for k, v in param_grid.items()}

        metrics = classifier_model(
            model,
            X_train[:, bfs], 
            y_train,
            X_test[:, bfs],
            y_test,
            printing=False,
            params=params
        )

        print("Parameters:", params)

        score = metrics["f1-score"]
        return params, score, [params, metrics]
    
def choose_average_bfs(all_metrics):
    param_bfs_all = {}
    for scores, metrics in all_metrics:
        for i in range(len(scores)):
            rank = f"{scores[:i+1]}"
            if rank in param_bfs_all:
                param_bfs_all[rank].append(metrics[i]['f1-score'])
            else:
                param_bfs_all[rank] = [metrics[i]['f1-score']]
    
    best_score = 0
    param_bfs = None
    for key, values in param_bfs_all.items():
        media = np.mean(values)
        if media > best_score:
            best_score = media
            param_bfs = key

    texto = param_bfs.strip("[]")
    # separa pelos espaços
    result = [int(x) for x in texto.split()]

    return result, best_score


def choose_average_params(all_metrics):
    param_bfs_all = {}
    for scores, metrics in all_metrics:
        rank = f"{scores}"
        if rank in param_bfs_all:
            param_bfs_all[rank].append(metrics)
        else:
            param_bfs_all[rank] = [metrics]
    
    best_score = 0
    param_bfs = None
    for key, values in param_bfs_all.items():
        media = np.mean(values)
        if media > best_score:
            best_score = media
            param_bfs = key

    result = ast.literal_eval(param_bfs)

    return result, best_score
    
def deployModel(X_train, y_train, X_test, y_test, model, bfs, parameters, filename, label = "",):
    
    clf = reset_model(model)

    # model, X_train, y_train, X_test, y_test, label = "", printing=True
    metrics = classifier_model(clf, X_train[:, bfs], y_train, X_test[:,bfs], y_test, label=label, printing=True)

    clf = reset_model(model)

    X_train = np.vstack((X_train, X_test))         # vertical stack
    y_train = np.concatenate((y_train, y_test))    # concatenate because 1 dimension

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

        dic_2 = {"X_train_orig": X_train_orig, 
            "y_train_orig": y_train_orig,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "X_val": X_val,
            "y_val": y_val}
        
        folds.append(dic_2)
        
    return folds

import numpy as np

def create_repeated_person_folds(
    X,
    y,
    person_col_index=-1,
    n_splits=10,
    n_repeats=10,
    random_state=42
):
    """
    Repeated cross-validation por pessoa (ex: 10x10 = 100 folds)

    Cada fold:
    - ~80% pessoas treino
    - ~10% validação
    - ~10% teste
    """

    rng = np.random.default_rng(random_state)
    persons = X[:, person_col_index].astype(int)
    unique_persons = np.unique(persons)

    n_persons = len(unique_persons)
    n_test = max(1, int(0.1 * n_persons))
    n_val  = max(1, int(0.1 * n_persons))

    folds = []

    for rep in range(n_repeats):
        # Baralhar pessoas a cada repetição
        shuffled = unique_persons.copy()
        rng.shuffle(shuffled)

        for fold in range(n_splits):
            # Rotação circular
            rotated = np.roll(shuffled, fold)

            test_persons = rotated[:n_test]
            val_persons  = rotated[n_test:n_test + n_val]
            train_persons = rotated[n_test + n_val:]

            train_mask = np.isin(persons, train_persons)
            val_mask   = np.isin(persons, val_persons)
            test_mask  = np.isin(persons, test_persons)

            X_feat = np.delete(X, person_col_index, axis=1)

             # NORMALIZAÇÃO
            scaler = MinMaxScaler()

            # Normaliza para os máximos e minimos do X_Train e guarda
            X_train_scaled = scaler.fit_transform(X_feat[train_mask])
            y_train = y[train_mask]

            # Normaliza com os máximos e minimos guardados
            X_val_scaled = scaler.transform(X_feat[val_mask])
            y_val = y[val_mask]

            X_test_scaled = scaler.transform(X_feat[test_mask])
            y_test = y[test_mask]

            X_train_orign = np.vstack((X_train_scaled, X_val_scaled))
            y_train_orign = np.concatenate((y_train, y_val), axis=0)


            folds.append({
                "X_train_orig": X_train_orign,
                "y_train_orig": y_train_orign,
                "X_train": X_train_scaled,
                "y_train": y_train,
                "X_val":   X_val_scaled,
                "y_val":   y_val,
                "X_test":  X_test_scaled,
                "y_test":  y_test,
                "rep": rep,
                "fold": fold,
                "train_persons": train_persons,
                "val_persons": val_persons,
                "test_persons": test_persons
            })

    print(f"Total de folds criados: {len(folds)}")
    return folds

def chooseModel(f1_all_folds, printing=True):
    """
    Recebe uma lista de dicionários com F1-scores por fold e modelo.
    Calcula média e desvio padrão de cada modelo, imprime num DataFrame,
    e devolve o modelo com maior média.
    """
    # Converter para DataFrame
    df_f1 = pd.DataFrame(f1_all_folds)
    
    # Calcular média e desvio padrão por modelo
    stats_df = pd.DataFrame({
        "Mean_F1": df_f1.mean(),
        "Std_F1": df_f1.std()
    })

    df_f1.to_csv("./graphics/cv_models.csv")
    
    # Mostrar
    if printing:
        display(stats_df)
    
    # Modelo com maior média
    best_model = stats_df["Mean_F1"].idxmax()
    print("Modelo com maior média de F1:", best_model)
    
    return best_model

######################################################################

def train_tvt(X, y, model, parameters, filename, random_state = SEED, label="", flagfeatureRanking = True, use_iris=True):
    dic = split_set(X, y, random_state=random_state, use_iris=use_iris)
    X_train = dic["X_train"]
    X_val = dic["X_val"]
    y_train = dic["y_train"]
    y_val = dic["y_val"]

    default_parameters = pick_first_param_values(parameters)

    scores = compute_feature_ranking(X_train, y_train, printing=True)

    if flagfeatureRanking:
        bfs, _ = featureRanking(X_train, y_train, X_val, y_val, model, scores, params=default_parameters, save=True, title="TVT", filename="./ElbowGraphs/iris/tvt/elbow_graph.png")
    else:
        bfs = scores
    best_parameters, _, _ = chooseParameters(X_train, y_train, X_val, y_val, model, bfs, parameters)
    metrics = deployModel(dic["X_train_orig"], dic["y_train_orig"], dic["X_test"], dic["y_test"], model, bfs, best_parameters, filename, label=label)
    return metrics

def train_cv(X, y, models, parameters, filename, random_state = SEED, n_folds = 10, n_repeats = 10, label="", flagfeatureRanking=True, use_iris=True, flagPrintingFoldNumber=True):
    # Se 'models' NÃO for um dicionário (ou seja, é um modelo solto)
    if not isinstance(models, dict):
        # 1. Descobrimos um nome automático (ex: "RandomForestClassifier")
        model_name = type(models).__name__ 
        
        # 2. Transformamos o modelo único num dicionário
        models = {model_name: models}
        
        # 3. CRÍTICO: Temos de fazer o mesmo com os 'parameters' 
        # para que a linha parameters[modelName] não falhe mais tarde
        if not isinstance(parameters, dict):
            parameters = {model_name: parameters}

    if use_iris==True:
        folds = createFolds(X, y, 10, 10)
    else: 
        folds = create_repeated_person_folds(X , y, person_col_index=-1, n_splits=10, n_repeats=10, random_state=SEED)

    f1_all_folds = []

    f = 0
    for fold in folds:
        if flagPrintingFoldNumber:
            print(f"Fold {f}")
        f+=1

        f1_this_fold_models = {model_name: None for model_name in models.keys()}

        X_train_orig = fold["X_train_orig"]
        y_train_orig = fold["y_train_orig"]
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        X_val = fold["X_val"]
        y_val = fold["y_val"]

        if flagfeatureRanking:
            scores = compute_feature_ranking(X_train, y_train, printing=False)
        else:
            scores = list(range(X.shape[1]))

        m = 0
        for modelName, model in models.items():            
            default_parameter = pick_first_param_values(parameters[modelName])

            if flagfeatureRanking:
                bfs, _ = featureRanking(X_train, y_train, X_val, y_val, model, scores, default_parameter, plot=False, printing=False, save=True, title=f"CV | {modelName} | fold {f}", filename=f"./ElbowGraphs/iris/cv_k_{model.n_neighbors}/fold_{f}_{modelName}.png")
                best_parameters, _, _= chooseParameters(X_train, y_train, X_val, y_val, model, bfs, parameters[modelName])
            else:
                bfs = scores
                best_parameters = default_parameter

            score = classifier_model(model, X_train_orig[:, bfs], y_train_orig, X_test[:, bfs], y_test, printing=False, params=best_parameters)["f1-score"]

            f1_this_fold_models[modelName] = score

            m += 1
        
        f1_all_folds.append(f1_this_fold_models)

    best_model = chooseModel(f1_all_folds, printing=True)

    print(models[best_model])

    return models[best_model], best_model, parameters[best_model]

def plot_metrics(metrics_list, metric_name='f1-score', title='Metric Plot'):
    ks = [m['k'] for m in metrics_list]
    values = [m[metric_name] for m in metrics_list]

    plt.figure(figsize=(6,4))
    plt.plot(ks, values, marker='o')
    plt.xlabel("k (n_neighbors)")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.grid(True)
    plt.show()

def deployment_cv(X, y, model, modelName, parameters, filename, random_state = SEED, n_folds = 10, n_repeats = 10, label="", flagPrintingFoldNumber=True, flagfeatureRanking=True):
    folds = createFolds(X, y, 10, 10)

    f1_all_folds = []

    bfs_metrics = []

    f = 0

    if flagfeatureRanking:
        for fold in folds:
            if flagPrintingFoldNumber:
                print(f"Fold {f}")
            f+=1

            X_train_orig = fold["X_train_orig"]
            y_train_orig = fold["y_train_orig"]
            X_test = fold["X_test"]
            y_test = fold["y_test"]
            
            scores = compute_feature_ranking(X_train_orig, y_train_orig, printing=False)
                    
            default_parameter = pick_first_param_values(parameters)

            bfs, metrics = featureRanking(X_train_orig, y_train_orig, X_test, y_test, model, scores, default_parameter, plot=False, printing=False, save=True, title=f"CV Deploy | Fold {f} | {modelName}", filename=f"./ElbowGraphs/iris/cv/deploy/fold_{f}_{modelName}.png")

            bfs_metrics.append([scores, metrics])
        bfs_final, bfs_score = choose_average_bfs(bfs_metrics)
        print(bfs_final, bfs_score)

        parameters_metrics = []
        for fold in folds:
            best_parameters, score, params_list = chooseParameters(X_train_orig, y_train_orig, X_test, y_test, model, bfs_final, parameters)

            parameters_metrics.append([best_parameters, score])

        print(parameters_metrics)

        best_parameters, _ = choose_average_params(parameters_metrics)
    else:
        bfs_final = list(range(X.shape[1]))
        best_parameters = pick_first_param_values(parameters)

    metrics = deployModel(X_train_orig, y_train_orig, X_test, y_test, model, bfs, best_parameters, filename, label=label)

    return metrics

def train_TO(X, y, model, printing=True, label="TO"):
    metrics = classifier_model(model, X, y, X, y, label=label, printing=printing)
    return metrics

def train_TT(X, y, model, printing=True, label="TT", random_state=SEED, use_iris=True):
    X_train, X_test, y_train, y_test = split_set(X, y, test_size=0.3, val_size=0, random_state=random_state, use_iris=use_iris)
    metrics = classifier_model(model, X_train, y_train, X_test, y_test, label=label, printing=printing)
    return metrics

def averageInCV(X, y, model, flagPrintingFoldNumber=False, use_iris=True):
    if use_iris==True:
        folds = createFolds(X, y, 10, 10)
    else: 
        folds = create_repeated_person_folds(X , y, person_col_index=-1, n_splits=10, n_repeats=10, random_state=SEED)

    f1_scores_by_folds = []
    recall_by_folds = []
    precision_by_folds = []

    f = 0
    for fold in folds:
        if flagPrintingFoldNumber:
            print(f"Fold {f}")
        f+=1

        X_train_orig = fold["X_train_orig"]
        y_train_orig = fold["y_train_orig"]
        X_train = fold["X_train"]
        y_train = fold["y_train"]
        X_test = fold["X_test"]
        y_test = fold["y_test"]
        X_val = fold["X_val"]
        y_val = fold["y_val"]

        metrics = classifier_model(model, X_train, y_train, X_test, y_test, label="", printing=False)

        f1_scores_by_folds.append(metrics["f1-score"])
        recall_by_folds.append(metrics["recall"])
        precision_by_folds.append(metrics["precision"])
    
    f1_mean = sum(f1_scores_by_folds) / len(f1_scores_by_folds)
    recall_mean = sum(recall_by_folds) / len(recall_by_folds)
    precision_mean = sum(precision_by_folds) / len(precision_by_folds)

    return f1_mean, recall_mean, precision_mean

def evaluate_with_kfold(X, y, classifier, rkf, label="KFOLD", printing=True):
    y_preds, y_trues = [], []

    for train_idx, test_idx in rkf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = clone(classifier)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_trues.append(y_test)
        y_preds.append(y_pred)

    # metrics_df = metrics_to_dataframe(y_trues, y_preds, label=label)
    metrics = print_metrics(y_trues, y_preds, label=label, printing=printing)
    return metrics

def run_model(X, y, model, split_scheme, parameters, filename, label="", random_state=SEED, use_iris=True, feature_ranking = True):
    if split_scheme == "TVT":
        return train_tvt(X, y, model, parameters, filename, random_state=random_state, label=label, use_iris=use_iris)
    elif split_scheme == "CV-base":
        f1_mean, recall_mean, precision_mean = averageInCV(X, y, model, flagPrintingFoldNumber=False, use_iris=use_iris)
        metrics = {"recall": recall_mean, "precision": precision_mean, "f1-score": f1_mean}
        print(f"\n===== {label} (means) =====")
        print(f"Recall:    {recall_mean:.4f}")
        print(f"Precision: {precision_mean:.4f}")
        print(f"F1-Score:  {f1_mean:.4f}")
        print("=========================")
        return metrics
    elif split_scheme == "CV":
        best_model, best_model_name, parameters = train_cv(X, y, model, parameters, filename, random_state=random_state, label=label, use_iris=use_iris, flagfeatureRanking=feature_ranking, flagPrintingFoldNumber=False)
        metrics = deployment_cv(X, y, best_model, best_model_name, parameters, filename, random_state=random_state, label=label, flagfeatureRanking=feature_ranking, flagPrintingFoldNumber=False)
        return metrics
    elif split_scheme == "TO":
        if use_iris == True:
            X = np.delete(X, -1, axis=1)
        metrics = train_TO(X, y, model, printing=True, label=label)
        return metrics
    elif split_scheme == "TT":
        metrics = train_TT(X, y, model, printing=True, label=label, use_iris=use_iris)
        return metrics

    return -1

################################################

def load_from_file(filename="data.npy", description="Dados"):
    """
    Carrega um ficheiro .npy guardado com `save_to_file`.
    """
    try:
        data = np.load(filename, allow_pickle=True)
        print(f"[OK] {description} carregados de '{filename}'.")
        return data
    except FileNotFoundError:
        print(f"[INFO] Ficheiro '{filename}' não encontrado. {description} será recalculado.")
        return None