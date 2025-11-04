from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd


#Usar critério 70/30 
def create_split_train_test(X, y, test_size=0.30, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)

    return X_train, X_test, y_train, y_test

#Usar critério 40/30/30
def create_split_tvt(X, y,val_size= test_size=0.30, random_state=42):
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