import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from modelTraining import precision, f1_score, recall, confusion_matrix, print_metrics
from sklearn.preprocessing import StandardScaler
from modelTraining import precision, recall, f1, print_metrics

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
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialização aleatória dos pesos (W) e bias (b)
        
        # Camada 1: Conecta Input -> Hidden
        # Formato: (Features de Entrada, Neurónios da Hidden)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01 
        self.b1 = np.zeros((1, hidden_size)) # O 1 permite somar a todas as linhas (broadcasting)

        # Camada 2: Conecta Hidden -> Output
        # Formato: (Neurónios da Hidden, Classes de Saída)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

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
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        if function_name == "tanh":
            return 1 - np.square(self.tanh(z))
        if function_name == "relu":
            y = (z > 0) * 1
            return y
        if function_name == "leaky_relu":
            return  np.where(z > 0, 1, 0.01)
        return "No such activation"

class OurNeuralNetwork:
    def __init__(self, layer_sizes, activation="relu", learning_rate=0.01):
        """
        layer_sizes: Lista com o tamanho de cada camada. Ex: [3, 5, 2] 
                     (3 inputs, 5 hidden, 2 outputs)
        """
        self.layer_sizes = layer_sizes
        self.activation_func = activation
        self.lr = learning_rate
        self.parameters = {}
        self.L = len(layer_sizes) - 1 # Número de camadas de conexões (pesos)
        
        # Inicialização dos Pesos (Formato: Samples x Features)
        for i in range(1, len(layer_sizes)):
            # Pesos: (Input_dim, Output_dim)
            # Ex: Camada 1 tem inputs da camada 0 e outputs da camada 1
            self.parameters[f"W{i}"] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01
            # Bias: (1, Output_dim)
            self.parameters[f"b{i}"] = np.zeros((1, layer_sizes[i]))
            
    # --- Funções de Ativação ---
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        # Axis=1 porque as classes estão nas colunas (para cada linha/amostra)
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def derivative(self, z, activation):
        if activation == "sigmoid":
            s = self.sigmoid(z)
            return s * (1 - s)
        elif activation == "relu":
            return (z > 0).astype(float)
        return 1.0 # Linear/Identity

    # --- Core Logic ---
    
    def forward(self, X):
        """
        Realiza a propagação para a frente.
        X shape: (N_amostras, N_features)
        """
        cache = {"A0": X} # A0 é o input
        
        # Loop pelas camadas escondidas
        for l in range(1, self.L):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            A_prev = cache[f"A{l-1}"]
            
            # MATH: X dot W (features nas colunas)
            Z = np.dot(A_prev, W) + b
            if self.activation_func == "sigmoid":
                A = self.sigmoid(Z)
            else:
                A = self.relu(Z)
                
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
            
        # Última camada (Output) - Geralmente usa Softmax para classificação
        W_last = self.parameters[f"W{self.L}"]
        b_last = self.parameters[f"b{self.L}"]
        A_last_prev = cache[f"A{self.L-1}"]
        
        Z_last = np.dot(A_last_prev, W_last) + b_last
        A_last = self.softmax(Z_last)
        
        cache[f"Z{self.L}"] = Z_last
        cache[f"A{self.L}"] = A_last
        
        return A_last, cache

    def backward(self, Y, cache):
        """
        Calcula os gradientes.
        Y deve ser one-hot encoded ou os labels.
        """
        grads = {}
        m = Y.shape[0] # Número de amostras
        
        # 1. Erro na saída (Predição - Real)
        # Assumindo Softmax com Cross-Entropy Loss
        A_final = cache[f"A{self.L}"]
        dZ = A_final - Y 
        
        # 2. Backpropagation reverso (da última camada para a primeira)
        for l in range(self.L, 0, -1):
            A_prev = cache[f"A{l-1}"]
            
            # Gradiente dos Pesos: (A_prev.T dot dZ) -> Inverso do site original!
            dW = (1/m) * np.dot(A_prev.T, dZ)
            # Gradiente do Bias: Soma nas colunas (axis=0) -> Inverso do site original!
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db
            
            # Se não for a primeira camada, calcula o erro para a camada anterior
            if l > 1:
                W = self.parameters[f"W{l}"]
                dZ_next = dZ
                Z_prev = cache[f"Z{l-1}"]
                
                # Math: Erro projectado para trás * Derivada da ativação
                dA = np.dot(dZ_next, W.T)
                dZ = dA * self.derivative(Z_prev, self.activation_func)

        return grads

    def update_parameters(self, grads):
        for l in range(1, self.L + 1):
            self.parameters[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def train(self, X, y, epochs=1000):
        # Se y não for one-hot, converter (simplificação)
        X = np.array(X)
        y = np.array(y)

        y = y.astype(int)
        # -------------------------------------------------------------

        # Se y não for one-hot, converter (simplificação)
        if y.ndim == 1 or y.shape[1] == 1:
            num_classes = self.layer_sizes[-1]
            # Agora o flatten() já vai funcionar porque y é um array numpy
            y_one_hot = np.eye(num_classes)[y.flatten()]
        else:
            y_one_hot = y
            
        loss_history = []
        
        for i in range(epochs):
            # 1. Forward
            A_final, cache = self.forward(X)
            
            # 2. Loss (Cross Entropy)
            m = X.shape[0]
            loss = -np.sum(y_one_hot * np.log(A_final + 1e-8)) / m
            loss_history.append(loss)
            
            # 3. Backward
            grads = self.backward(y_one_hot, cache)
            
            # 4. Update
            self.update_parameters(grads)
            
            if i % 100 == 0:
                print(f"Epoch {i}: Loss {loss:.4f}")
                
        return loss_history

    def predict(self, X):
        A_final, _ = self.forward(X)
        return np.argmax(A_final, axis=1) # Axis 1 = Classe com maior prob por linha
    
    def evaluate(self, X_test, y_test, label="Test Results"):
        """
        Faz a previsão e imprime todas as métricas automaticamente.
        """
        # 1. Fazer previsões
        y_pred = self.predict(X_test)
        
        # 2. Garantir que y_test está no formato correto (inteiros)
        if hasattr(y_test, 'values'):
            y_test = y_test.values.astype(int)
        else:
            y_test = np.array(y_test).astype(int)
            
        # 3. Chamar a função print_metrics (que já definimos acima)
        metrics = print_metrics(y_test, y_pred, label=label)
        
        return metrics