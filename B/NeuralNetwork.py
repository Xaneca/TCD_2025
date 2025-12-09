import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from modelTraining import precision, f1_score, recall, confusion_matrix, print_metrics
from sklearn.preprocessing import StandardScaler

SEED = 42

def createMLP(X_train, y_train, X_test, y_test, label="MLP", hidden_layers=(64,32), activation='relu',
              learning_rate_type='constant', learning_rate='0.01', momentum=0.0,
              max_iter=500, batch_size='auto', random_state=SEED):
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
    y_pred = model.predict(X_test, y_test)

    metrics = print_metrics(y_test, y_pred, label=label, printing=True)

    return metrics