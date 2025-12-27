import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from modelTraining import precision, f1_score, recall, confusion_matrix, print_metrics
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from tqdm import tqdm # To visualize progress bars during training

SEED = 42

def createMLP(X_train, y_train, X_test, y_test, label="MLP", hidden_layers=(64,), activation='relu',
              learning_rate_type='constant', learning_rate=0.01, momentum=0.0,
              max_iter=500, batch_size='auto', random_state=SEED, printing=True):
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,   # duas camadas escondidas
        activation=activation,
        solver='sgd',                  # obrigatório se usares learning rate controlado
        learning_rate=learning_rate_type,      # FIXO
        learning_rate_init=learning_rate,       # α fixo
        momentum=momentum,                  # sem momentum
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = print_metrics(y_test, y_pred, label=label, printing=printing)

    return metrics

import matplotlib.pyplot as plt

def createMLP_variableSizes(X_train, y_train, X_test, y_test, 
                            sizes_to_test=[10, 25, 50, 75, 100, 150, 200],
                            learning_rate_type='constant',
                            learning_rate=0.01, 
                            momentum=0.0,
                            max_iter=1000, 
                            random_state=SEED):
    """
    Treina várias MLPs com diferentes tamanhos de camada escondida (1 camada)
    e gera um gráfico comparativo de Precision, Recall e F1-Score.
    """
    
    # Listas para guardar os resultados
    list_precision = []
    list_recall = []
    list_f1 = []

    print(f"{'Neurónios':<10} | {'F1-Score':<10}")
    print("-" * 25)

    # Loop pelos tamanhos
    for size in sizes_to_test:
        
        # Chama a função createMLP original
        # Nota: printing=False para não encher a consola
        metrics = createMLP(
            X_train, y_train, X_test, y_test, 
            label=f"Size_{size}",
            hidden_layers=(size,),  # Cria a tupla de 1 camada
            learning_rate_type=learning_rate_type,
            learning_rate=learning_rate,
            momentum=momentum,
            max_iter=max_iter,
            random_state=random_state,
            printing=False
        )
        
        # Extrai métricas (com segurança caso a chave mude)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        # Tenta 'f1-score', se não existir tenta 'f1'
        f1 = metrics.get('f1-score', metrics.get('f1', 0))
        
        list_precision.append(prec) 
        list_recall.append(rec)
        list_f1.append(f1)
        
        print(f"{size:<10} | {f1:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    plt.plot(sizes_to_test, list_precision, marker='o', linestyle='--', color='blue', label='Precision')
    plt.plot(sizes_to_test, list_recall, marker='s', linestyle='--', color='green', label='Recall')
    plt.plot(sizes_to_test, list_f1, marker='^', linewidth=2, color='red', label='F1-Score')

    plt.title('Performance por Nº de Neurónios (1 Camada Escondida)')
    plt.xlabel('Número de Neurónios')
    plt.ylabel('Score (0.0 a 1.0)')
    plt.xticks(sizes_to_test)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.05)
    
    plt.show()
    
    # Retorna os dados caso queiras usar depois
    return {
        'sizes': sizes_to_test,
        'f1_scores': list_f1,
        'precision': list_precision,
        'recall': list_recall
    }

#######################################
############# OUR NETWORK #############
#######################################

def createOurMLP(data, labels):
    # select random to take its shape
    n = np.random.choice(np.arange(data.shape[0]+1))
    
class OurMLP:
    def __init__(self, data, labels):
        return
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)
    def leaky_relu(z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, z * 0.01)
    def softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z))
        return e / np.sum(e, axis=0)
    def normalize(x: np.ndarray) -> np.ndarray:
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    def one_hot_encode(x: np.ndarray, num_labels: int) -> np.ndarray:
        return np.eye(num_labels)[x]
    def derivative(function_name: str, z: np.ndarray) -> np.ndarray:
        if function_name == "sigmoid":
            return sigmoid(z) * (1 - sigmoid(z))
        if function_name == "tanh":
            return 1 - np.square(tanh(z))
        if function_name == "relu":
            y = (z > 0) * 1
            return y
        if function_name == "leaky_relu":
            return  np.where(z > 0, 1, 0.01)
        return "No such activation"