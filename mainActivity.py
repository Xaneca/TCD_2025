import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis, iqr, pearsonr
from numpy.fft import fft, fftfreq
from features import extract_feature_vector



PATH = "./../FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master"
PLOT_PATH = "./plots"
activities = pd.read_csv("nameActivities.csv")
titles_sensors = ["Left wrist", "Right wrist", "Chest", "Upper right leg", "Lower left leg"]
titles_vectors = ["Accelerometer", "Gyroscope", "Magnetometer"]
NUM_PEOPLE = 15
NUM_SENSORS = 5
NUM_ACTIVITIES = 16
NUM_COLUNAS = 12
FS = 51.2

#EX 4.2
SAMPLING_RATE = 100  # Hz [6]
WINDOW_LENGTH = 200  # 2 segundos * 100 Hz [5]
GRAVITY_G_UNITS = 1.0  # Aproximando a gravidade como 1g (assumindo que os dados de aceleração são em unidades de 'g')


individuals = []    # tam 15 -> tam 5 -> tam 12 + data
                    # (15, 5, 12)
sensors_data = []

people_data = []

def getFiles(path):
    for i in range(NUM_PEOPLE):
        ind = []
        for j in range(NUM_SENSORS):
            df = pd.read_csv(f'{path}/part{i}/part{i}dev{j + 1}.csv')
            ind.append(df.to_numpy())
        individuals.append(ind)
    return

def getIndividual(path, ind_num):
    ind = []
    for j in range(NUM_SENSORS):
        df = pd.read_csv(f'{path}/part{ind_num}/part{ind_num}dev{j + 1}.csv')
        ind.append(df.to_numpy())
    print("Individuals Extracted")
    return ind

# def calculateModule(ind, sensor_num, i_x, i_z):
#     # idx - 1,2,3 - accelerometer
#     # idx - 4,5,6 - gyroscope
#     # idx - 7,8,9 - magnetometer
#     return np.sqrt(np.sum(ind[sensor_num][:, i_x:i_z]**2, axis=1))   # axis = 1 -> por linha

def calculateModule(ind_sensor, i_x, i_z):
    # idx - 2,3,4  - accelerometer
    # idx - 5,6,7  - gyroscope
    # idx - 8,9,10 - magnetometer
    return np.sqrt(np.sum(ind_sensor[:, i_x:i_z + 1]**2, axis=1))   # axis = 1 -> por linha

def boxPlot_modules(plot = True):
    outliers = []
    for activity in range(1, NUM_ACTIVITIES + 1):
        this_activity_outliers = 0
        plt.figure(figsize =(10, 7))

        plt.title(f"{activities.loc[activity - 1, 'name']}")
        for sensor in range(NUM_SENSORS):
            all_people_data = np.empty((0, NUM_COLUNAS))  # inicializa array vazio com 0 linhas
            for ind in range(NUM_PEOPLE):
                arr = individuals[ind][sensor]  # array do indivíduo e sensor atual
                filtered = arr[arr[:, -1] == activity]
                all_people_data = np.vstack((all_people_data, filtered))

            for i in range(3):
                plt.subplot(3, 5, i * 5 + sensor + 1)
                bp = plt.boxplot(calculateModule(all_people_data, 1 + 3 * i, 3 + 3 * i))
                outl = bp['fliers'][0].get_ydata()  # bp['fliers'] is a Line2D list
                this_activity_outliers += len(np.unique(outl))
        if plot:
            plt.show()

        plt.close()
        outliers.append({
            "num_outliers": this_activity_outliers / 3,
            "num_points": all_people_data.shape[0]  # número de linhas
        })

    return outliers

def boxPlot_modules_2(plot = True):
    outliers = []
    titles = ["accelerometer", "gyroscope", "magnetometer"]

    #Junta todos os arrays num só array 
    new_list = []
    for j in range(NUM_PEOPLE):
        person = np.vstack(individuals[j])
        new_list.append(person)
    all_the_data = np.vstack(new_list)
    #print(all_the_data.shape)

    
    for i in range (3):        
        this_activity_outliers = 0
        
        #Cria os arrays com os valores para y e x
        y_vals = calculateModule(all_the_data, 1 + 3 * i, 3 + 3 * i) #Tira o modulo para um dos vectores
        #print(y_vals)
        x_vals = all_the_data[:, 11] #Retira todos os valores de x
        #print(x_vals)
        unique_x_vals = np.unique(x_vals) #Vai ver os valores unicos das atividades todas - podemos apenas fazer um array de 1 a 12 para poupar tempo -ver Xana

        #print(len(unique_x_vals))
        #Vamos agrupar os valores pelo seu x
        data_per_x = [y_vals[x_vals == x] for x in unique_x_vals]
        #print(len(data_per_x))

        #Imprime o boxplot
        plt.figure(figsize =(10, 7))
        plt.title(f"Boxplots de atividades para o modulo do vetor {titles[i]}")
        bp = plt.boxplot(data_per_x, positions=unique_x_vals, widths=0.6)
        

        plt.xticks(unique_x_vals, [int(x) for x in unique_x_vals]) #Força os valores de x a aparecerem como interiros
        plt.xlabel("Atividade")
        plt.ylabel(f"Valor do modulo do {titles[i]}")
        plt.grid(True)
        plt.show()


def boxPlot_modules_3(plot = True):
    outliers = [] #Lista que vai receber o número de outliers de todos os sensores e atividades

    #Junta todos os arrays num só array 
    for k in range (NUM_SENSORS):
        this_sensor_outliers = []
        new_list = []
        for j in range(NUM_PEOPLE):
            new_list.append(individuals[j][k])
        all_the_data = np.vstack(new_list)

        num_vectors = 3
        for i in range (num_vectors):        
            this_vector_outliers = []

            #Cria os arrays com os valores para y e x
            y_vals = calculateModule(all_the_data, 1 + 3 * i, 3 + 3 * i) #Tira o modulo para um dos vectores
            #print(y_vals)
            x_vals = all_the_data[:, 11] #Retira todos os valores de x - num das atividades
            #print(x_vals)
            unique_x_vals = np.unique(x_vals) #Vai ver os valores unicos das atividades todas - podemos apenas fazer um array de 1 a 16 para poupar tempo -ver Xana
                                                                                            # -> acho q nao pq é preciso associar cada valor de y a um x

            #Vamos agrupar os valores pelo seu x
            data_per_x = [y_vals[x_vals == x] for x in unique_x_vals]

            #Imprime o boxplot
            plt.figure(figsize =(10, 7))
            plt.title(f"Activity Boxplots for the Vector Module {titles_vectors[i]} of the {titles_sensors[k]} sensor")
            bp = plt.boxplot(data_per_x, positions=unique_x_vals, widths=0.6)
            for l, flier in enumerate(bp['fliers']):
                temp = []
                outliers_values = flier.get_ydata()
                temp.append(len(np.unique(outliers_values))) #Coloca o número de outliers sem duplicados
                temp.append(len(data_per_x[l])) #Coloca o número total de valores por cada boxplot para cada atividade
                this_vector_outliers.append(temp)

            plt.xticks(unique_x_vals, [int(x) for x in unique_x_vals]) #Força os valores de x a aparecerem como interiros
            plt.xlabel("Activity")
            plt.ylabel(f"Value of the Vector Module {titles_vectors[i]}")
            plt.grid(True)
            plt.savefig(PLOT_PATH + "/ex3_1" + f"/sensor{k}_vector{i}.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace

            if plot:
                plt.show()
            plt.close()
            this_sensor_outliers.append(this_vector_outliers)
        outliers.append(this_sensor_outliers)
    return outliers

def create_list_by_sensor():
    #all_by_sensor = []
    for k in range (NUM_SENSORS):
        new_list = []
        for j in range(NUM_PEOPLE):
            new_list.append(individuals[j][k])
        all_the_data = np.vstack(new_list)
        sensors_data.append(all_the_data) 

def create_list_complete():
    for j in range(NUM_PEOPLE):
        new_list = []
        for s in range (NUM_SENSORS):
            new_list.append(individuals[j][s])
        all_the_data = np.vstack(new_list)
        people_data.append(all_the_data)

def density(num_outliers, num_points):
    return num_outliers / num_points * 100

def calculateDensityOutliers(num_outliers_per_activity):
    print("------- DENSITY ------")
    for s in range(NUM_SENSORS):
        print(f"--------- Sensor {titles_sensors[s]} ---------")
        for v in range(3):
            print(f"  --------- Vector {titles_vectors[v]} ---------")
            for a in range(NUM_ACTIVITIES):
                d = density(num_outliers_per_activity[s][v][a][0], num_outliers_per_activity[s][v][a][1])
                print(f"    {activities.loc[a, 'name']}: {d:.2f}%")
    return

def z_scores(data, k):
    mean = np.mean(data)
    std_dev = np.std(data)

    z = (data - mean)/std_dev

    outliers_mask = np.abs(z) > k
    #not_outliers_mask = np.abs(z) <= k

    return outliers_mask

def show_outliers(start_idx, k, title):
    for s in range(NUM_SENSORS):
        all_x, all_y, all_colors = [], [], []
        for i in range(NUM_PEOPLE):
            module = calculateModule(individuals[i][s], start_idx, start_idx + 2)
            all_x.extend(individuals[i][s][:, -1])
            all_y.extend(module)
        outliers_mask = z_scores(all_y, k)
        all_colors.extend(np.where(outliers_mask, "red", "blue"))
        plt.figure(figsize = (8, 5))
        plt.scatter(all_x, all_y, c=all_colors, s=10, alpha=0.6)
        plt.xticks(range(1, NUM_ACTIVITIES + 1))
        plt.title(f"{title} for Sensor {titles_sensors[s]}")
        plt.xlabel("Activity")
        plt.ylabel(f"Value of the Vector Module {title}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(PLOT_PATH + "/ex3_4" + f"/sensor{s}_{title}.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace
        plt.show()
    return

def show_outliers(start_idx, k, title, plot = True):
    sensors_list = create_list_by_sensor()
    for s in range(NUM_SENSORS):
        print(f"\tSensor {titles_sensors[s]}")
        colors = []
        this_sensor = sensors_list[s]
        y = calculateModule(this_sensor, start_idx, start_idx + 2)
        x = this_sensor[:,-1]   # activity value
        outliers_mask = z_scores(y, k)

        for act in np.unique(x):
            mask = x == act
            y_act = y[mask]
            mean = np.mean(y_act)
            std = np.std(y_act)

            if std != 0:
                z = (y_act - mean) / std
                outliers_mask[mask] = np.abs(z) > k  # marca só os dessa atividade

        colors.extend(np.where(outliers_mask, "red", "blue"))
        #print(colors[:50])
        plt.figure(figsize = (8,5))
        plt.scatter(x, y, c=colors, s=10, alpha=0.6)
        plt.xticks(range(1, NUM_ACTIVITIES + 1))
        plt.title(f"{title} for Sensor {titles_sensors[s]}")
        plt.xlabel("Activity")
        plt.ylabel(f"Value of the Vector Module {title}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(PLOT_PATH + "/ex3_4" + f"/sensor{s}_{title}.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace
        if plot:
            plt.show()
        plt.close()

    return

def ex_3_4(k, plot = True):
    # accelerometter:
    print(f"Vector {titles_vectors[0]}:")
    show_outliers(1, k, titles_vectors[0], plot)

    # gyroscope
    print(f"Vector {titles_vectors[1]}:")
    show_outliers(4, k, titles_vectors[1], plot)

    # mangetometer
    print(f"Vector {titles_vectors[2]}:")
    show_outliers(7, k, titles_vectors[2], plot)

    return

#Input da função (dados em nparry, number of clusters, maximo de iterações, limite, )
def kmeans(X, n_clusters, max_iters, tol, random_state=None):
    # Configura semente aleatória
    if random_state is not None:
        np.random.seed(random_state)

    # Escolhe aleatoriamente centróides iniciais
    indices = np.random.choice(X.shape[0], n_clusters, replace=False) #Escolhe n_clusters do array X e oreplace = False faz com que não escolha pontos do array X repetidos
    centroids = X[indices] 

    for iteration in range(max_iters):
        # Atribui cada ponto ao centróide mais próximo
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Calcula novos centróides
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])

        # Verifica convergência
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"Convergência atingida na iteração {iteration + 1}")
            break

        centroids = new_centroids

    return centroids, labels, distances

def graph_3d(centroids, data, labels, outliers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extrair coordenadas dos dados
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

   # Máscaras
    mask_outliers = outliers[:, 1] == 1
    mask_normais  = outliers[:, 1] == 0

    # Plotar pontos normais com as cores dos labels
    scatter = ax.scatter(
        x[mask_normais],
        y[mask_normais],
        z[mask_normais],
        c=labels[mask_normais],
        cmap='tab10',
        s=10,
        alpha=0.6,
    )

    # Plotar outliers a cinzento
    ax.scatter(
        x[mask_outliers],
        y[mask_outliers],
        z[mask_outliers],
        color='gray',
        s=15,           # podes aumentar um pouco o tamanho para destacar
        alpha=0.8,
        label='Outliers'
    )


    # Adicionar legenda automática (opcional)
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    # Plotar os centroides a preto, com marcador diferente e maior
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
               c='black', s=200, marker='X', label='Centroides')

    # Rótulos dos eixos
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')

    # Mostrar legenda
    ax.legend()

    plt.show()

def calculate_outliers_centroids(distances, k, labels, limit):

    # Retira o valor minimo de cada linha de distances e mantém no formato 2d ou seja (n_amostras, 1) com o keepdims
    distance_from_centroid = np.min(distances, axis=1, keepdims=True) 

    # 1 se maior que y, 0 caso contrário
    sinalizacao = (distance_from_centroid > limit).astype(int)

    outliers = np.hstack([distance_from_centroid, sinalizacao])
    print(outliers)

    n_outliers = np.sum(outliers[:, 1] == 1)

    return n_outliers, outliers

def calculate_outliers_by_centroids(distances, k, labels):

    # Distância de cada ponto ao seu centróide
    distance_from_centroid = np.min(distances, axis=1, keepdims=True)

    # Inicializa o array de sinalização
    sinalizacao = np.zeros_like(distance_from_centroid, dtype=int)

    # Calcula limites por centróide
    for c in np.unique(labels):
        # Distâncias dos pontos do centróide c
        mask = labels == c #cria uma mascara booleana para sinalizar a true todos os pontos do centroid c
        dist_c = distance_from_centroid[mask] #faz um array só com as distancias dos pontos daquele centroid àquele centroid

        # Média e desvio padrão
        mean_c = np.mean(dist_c)
        std_c = np.std(dist_c)

        # Limite para considerar outlier (média + k * std)
        limit_c = mean_c + k * std_c

        # Marca como outlier (1) se maior que limite
        sinalizacao[mask] = (dist_c > limit_c).astype(int)

    # Junta distância + flag
    outliers = np.hstack([distance_from_centroid, sinalizacao])

    # Conta número total de outliers
    n_outliers = np.sum(sinalizacao)

    return n_outliers, outliers

#EX 3.8
def inject_outliers(data, k, percentage, z):
    data = np.array(data, dtype=float)
    mean = np.mean(data)
    std_dev = np.std(data)

    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev

    # Identificar outliers atuais
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    current_outlier_indices = np.where(outlier_mask)[0]
    current_outliers = len(current_outlier_indices)
    total = len(data)
    current_density = current_outliers / total * 100
    #density(current_outliers, total)   #############################################################

    # Se já há outliers suficientes → termina
    if current_density >= percentage:
        return data, current_outliers, current_outlier_indices

    # Caso contrário, injetar novos outliers
    target_outliers = int(percentage/100 * total)
    print(current_outliers)
    print(target_outliers)
    n_to_inject = target_outliers - current_outliers
    print(n_to_inject)

    # Escolher índices aleatórios que não sejam já outliers
    candidate_indices = np.where(~outlier_mask)[0]
    candidate_indices = candidate_indices[candidate_indices >= 100]
    np.random.shuffle(candidate_indices)
    inject_indices = candidate_indices[:n_to_inject]
    print(inject_indices)

    # s ∈ {-1, +1}
    s = np.random.choice([-1, 1], size=n_to_inject)
    print(s.shape)

    # q ∈ [0, z)
    q = np.random.uniform(0, z, size=n_to_inject)
    print(q.shape)

    # Aplicar fórmula: μ + s × k × (σ + q)
    new_values = mean + s * k * (std_dev + q)
    print(new_values.shape)

    # Injetar os novos valores
    new_values_1d = new_values.ravel()
    data[inject_indices] = new_values_1d.reshape(-1, 1)
    print(data[inject_indices].shape)

    all_outlier_indices = np.sort(
        np.concatenate((current_outlier_indices, inject_indices))
    )

    return data, target_outliers, all_outlier_indices

def inject_outliers_centered(data, k, percentage, z):
    original_data = data
    data = np.array(data[:, 0], dtype=float)
    mean = np.mean(data)
    std_dev = np.std(data)

    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev

    # Identificar outliers atuais
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    current_outlier_indices = np.where(outlier_mask)[0]
    current_outliers = len(current_outlier_indices)
    total = len(data)
    current_density = current_outliers / total * 100
    #density(current_outliers, total)   #############################################################

    # Se já há outliers suficientes → termina
    if current_density >= percentage:
        return data, current_outliers, current_outlier_indices

    # Caso contrário, injetar novos outliers
    target_outliers = int(percentage/100 * total)
    print(current_outliers)
    print(target_outliers)
    n_to_inject = target_outliers - current_outliers
    print(n_to_inject)

    # Escolher índices aleatórios que não sejam já outliers
    candidate_indices = np.where(~outlier_mask)[0]
    size = data.shape[0]
    candidate_indices = candidate_indices[(candidate_indices >= 50) & (candidate_indices < size - 50)]
    np.random.shuffle(candidate_indices)
    inject_indices = candidate_indices[:n_to_inject]
    print(inject_indices)

    # s ∈ {-1, +1}
    s = np.random.choice([-1, 1], size=n_to_inject)
    print(s.shape)

    # q ∈ [0, z)
    q = np.random.uniform(0, z, size=n_to_inject)
    print(q.shape)

    # Aplicar fórmula: μ + s × k × (σ + q)
    new_values = mean + s * k * (std_dev + q)
    print(new_values.shape)

    # Injetar os novos valores
    new_values_1d = new_values.ravel()
    data[inject_indices] = new_values_1d
    print(data[inject_indices].shape)

    all_outlier_indices = np.sort(
        np.concatenate((current_outlier_indices, inject_indices))
    )

    original_data[:, 0] = data

    return original_data, target_outliers, all_outlier_indices

# EX3.9
def linear_model(X, y):
    # Adicionar a coluna de 1s para o intercepto (β0)
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Calcular betas com a fórmula (XᵀX)⁻¹ Xᵀ y
    beta = np.linalg.pinv(X) @ y

    return beta

def linear_model_2(X, y):
    # Cria o modelo linear
    model = LinearRegression(fit_intercept=True)
    
    # Treina o modelo (ajusta os betas)
    model.fit(X, y)
    
    return model


def linear_model_predict_2(X, model):
    y_pred = model.predict(X)
    return y_pred

#EX3.9
def linear_model_predict(X, beta):
    X = np.column_stack((np.ones(X.shape[0]), X))  # intercepto
    #preve o y pelo modelo linear
    y_pred = X @ beta

    return y_pred


def generate_windows(data, p):
    # Garantir que data tem 2 dimensões
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    x, y = [], []
    for i in range(len(data) - p):
        # Janela de entrada
        x.append(np.hstack(data[i:i+p, :]))  # usa p valores consecutivos
        # Valor a prever (seguinte)
        y.append(data[i+p, :])

    #print(x)
    #print(y)

    # Converter listas em arrays numpy
    return np.vstack(x), np.vstack(y)

def cross_validation(data, p_values, n_splits=5):
    errors = []

    for p in p_values:
        X, y = generate_windows(data, p)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_errors = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            beta = linear_model(X_train, y_train)
            y_pred = linear_model_predict(X_test, beta)

            mse = mean_squared_error(y_test, y_pred)
            fold_errors.append(mse)

        mean_error = np.mean(fold_errors)
        errors.append(mean_error)
        print(f"p = {p}, erro médio = {mean_error:.4f}")

    return errors

def cross_validation_3_11(data, p_values, n_splits=5):
    errors = []

    for p in p_values:
        X, y = generate_centered_windows(data, p)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_errors = []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            beta = linear_model(X_train, y_train)
            y_pred = linear_model_predict(X_test, beta)

            mse = mean_squared_error(y_test, y_pred)
            fold_errors.append(mse)

        mean_error = np.mean(fold_errors)
        errors.append(mean_error)
        print(f"p = {p}, erro médio = {mean_error:.4f}")

    return errors

def generate_windows_outliers(data, outliers_indexes, p):
    # Garantir que data tem 2 dimensões
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    x, y = [], []
    for index in outliers_indexes:
        # Janela de entrada
        x.append(data[index-p:index, :].flatten())  # usa p valores consecutivos
        # Valor a prever (seguinte)
        y.append(data[index, :])

    #print(x)
    #print(y)

    # Converter listas em arrays numpy
    return np.vstack(x), np.vstack(y)

def removes_outliers_for_predictions(data, outlier_indices, y_prev):
    i = 0
    for index in outlier_indices:
        data[index] = y_prev[i]
        i += 1
    return data

def removes_outliers_for_predictions_2(data, outlier_indices, y_prev):
    data[outlier_indices] = y_prev
    return data, y_prev

def plot_outlier_replacement(data, outlier_indices, y_prev):
    # Garantir que os dados são arrays NumPy
    data = np.array(data)
    y_prev = np.array(y_prev)
    outlier_indices = np.array(outlier_indices)

    # ===== Gráfico 1: Outliers =====
    plt.figure(figsize=(10, 5))
    plt.scatter(outlier_indices, data[outlier_indices],
                color='red', label="Outliers", s=5, zorder=3)
    plt.title("Outliers no Sinal")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===== Gráfico 2: Valores Substituídos =====
    plt.figure(figsize=(10, 5))
    plt.scatter(outlier_indices, y_prev,
                color='yellow', label="Valores previstos (y_prev)", s=5, marker='o', zorder=3)
    plt.title("Valores Substituídos (y_prev)")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===== Gráfico 3: Sinal Original =====
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Dados originais", color="blue", linewidth=1, zorder=3)
    plt.title("Sinal Original Completo")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_outlier_replacement_3_11(data, y, outlier_indices, y_prev):
    # Garantir que os dados são arrays NumPy
    y = np.array(y)
    print(y.shape)
    y_prev = np.array(y_prev)
    print(y_prev.shape)
    outlier_indices = np.array(outlier_indices)
    print(outlier_indices.shape)

    # ===== Gráfico 1: Outliers =====
    plt.figure(figsize=(10, 5))
    plt.scatter(outlier_indices, y,
                color='red', label="Outliers", s=5, zorder=3)
    plt.title("Outliers no Sinal")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===== Gráfico 2: Valores Substituídos =====
    plt.figure(figsize=(10, 5))
    plt.scatter(outlier_indices, y_prev,
                color='yellow', label="Valores previstos (y_prev)", s=5, marker='o', zorder=3)
    plt.title("Valores Substituídos (y_prev)")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ===== Gráfico 3: Sinal Original =====
    plt.figure(figsize=(10, 5))
    plt.plot(data[:, 0], label="Dados originais", color="blue", linewidth=1, zorder=3)
    plt.title("Sinal Original Completo")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_outlier_replacement_together(data, outlier_indices, y_prev):
    # Garantir que os dados são arrays NumPy
    data = np.array(data)
    y_prev = np.array(y_prev)
    outlier_indices = np.array(outlier_indices)

    # Plot do sinal original
    plt.figure(figsize=(20, 10))

    # Plot dos outliers
    plt.scatter(outlier_indices, data[outlier_indices],
                color='red', label="Outliers", s=60, zorder=3)

    # Plot dos valores substitutos
    plt.scatter(outlier_indices, y_prev,
                color='yellow', label="Valores previstos (y_prev)", s=60, marker='x', zorder=4)

    # Estética
    plt.title("Comparação: Outliers vs Valores Substituídos")
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

def ex_3_10(modules_acceleration):

    '''
    t = np.linspace(0, 50, 1000)  # 1000 pontos entre 0 e 50
    data = np.sin(t) + 0.1 * np.random.randn(len(t))  # seno + ruído pequeno
    data = data.reshape(-1, 1)  # garantir formato (N, 1) #um teste para saber se o linear model estava correcto e está'''

    p_values = range(1,101)

    #errors = cross_validation(modules_acceleration, p_values, 5)

    errors = np.array([
    1.2311, 1.1998, 1.1992, 1.1976, 1.1971, 1.1913, 1.1809, 1.1745, 1.1714, 1.1698,
    1.1686, 1.1676, 1.1670, 1.1664, 1.1660, 1.1659, 1.1659, 1.1659, 1.1659, 1.1659,
    1.1659, 1.1658, 1.1656, 1.1653, 1.1649, 1.1645, 1.1640, 1.1633, 1.1622, 1.1608,
    1.1592, 1.1574, 1.1559, 1.1549, 1.1542, 1.1536, 1.1531, 1.1526, 1.1519, 1.1513,
    1.1508, 1.1502, 1.1499, 1.1497, 1.1495, 1.1494, 1.1494, 1.1494, 1.1494, 1.1493,
    1.1493, 1.1492, 1.1492, 1.1489, 1.1487, 1.1486, 1.1481, 1.1473, 1.1462, 1.1452,
    1.1447, 1.1440, 1.1431, 1.1417, 1.1402, 1.1390, 1.1381, 1.1370, 1.1363, 1.1357,
    1.1353, 1.1348, 1.1338, 1.1329, 1.1319, 1.1309, 1.1298, 1.1290, 1.1280, 1.1270,
    1.1260, 1.1251, 1.1244, 1.1239, 1.1232, 1.1227, 1.1223, 1.1218, 1.1214, 1.1209,
    1.1205, 1.1203, 1.1201, 1.1199, 1.1197, 1.1196, 1.1195, 1.1194, 1.1194, 1.1193
    ])

    plt.figure(figsize=(8,5))
    plt.plot(p_values, errors, marker='o')
    plt.xlabel('Valor de p')
    plt.ylabel('Erro médio (MSE)')
    plt.title('Erro de previsão vs Tamanho da janela (p)')
    plt.grid(True)
    plt.show()
    plt.savefig(PLOT_PATH + "/ex3_10" + f"/crossoverValidation.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace


    k = 2 #Valor normalmente usado para k
    data, target_outliers, outlier_indices = inject_outliers(modules_acceleration, k, 10, 1)

    # Para esta iteração escolher o p que tiver menos erro no cross_validation depois de rodar o código pelo gráfico e valores do erro médio decidiu-se usar o valor de p a 100
    x, y = generate_windows(modules_acceleration, 100)
    beta = linear_model(x, y)

    np.sort(outlier_indices)
    x1, y1 = generate_windows_outliers(modules_acceleration, outlier_indices, 100)
    y_prev = linear_model_predict(x1, beta)

    modules_acceleration_no_outliers = removes_outliers_for_predictions(modules_acceleration, outlier_indices, y_prev)

    plot_outlier_replacement(data, outlier_indices, y_prev)

    return

#EX3.11
def generate_centered_windows(data, p):
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    half_p = p // 2
    x, y = [], []

    for i in range(half_p, len(data) - half_p):
        # p/2 valores antes e p/2 depois
        window = np.vstack((
            data[i - half_p:i, :],
            data[i + 1:i + half_p + 1, :]
        ))
        window = np.hstack(window)
        x.append(window)
        y.append(data[i, 0])

    return np.vstack(x), np.vstack(y)

def generate_centered_windows_outliers(data, outliers_indexes, p):
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    half_p = p // 2
    x, y = [], []

    for index in outliers_indexes:
        # p/2 valores antes e p/2 depois
        window = np.vstack((
            data[index - half_p:index, :],
            data[index + 1:index + half_p + 1, :]
        ))
        window = np.hstack(window)
        x.append(window)
        y.append(data[index, 0])

    return np.vstack(x), np.vstack(y)

def ex_3_11(all_modules):

    p_values = range(10, 61, 10)

    # errors = cross_validation_3_11(all_modules, p_values, 5)

    errors = np.array([
    0.6183, 0.6164, 0.6159, 0.6156, 0.6156, 0.6155
    ])

    plt.figure(figsize=(8,5))
    plt.plot(p_values, errors, marker='o')
    plt.xlabel('Valor de p')
    plt.ylabel('Erro médio (MSE)')
    plt.title('Erro de previsão vs Tamanho da janela (p)')
    plt.grid(True)
    plt.show()
    plt.savefig(PLOT_PATH + "/ex3_10" + f"/crossoverValidation.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace


    k = 2 #Valor normalmente usado para k
    data, target_outliers, outlier_indices = inject_outliers_centered(all_modules, k, 10, 1)

    # Para esta iteração escolher o p que tiver menos erro no cross_validation depois de rodar o código pelo gráfico e valores do erro médio decidiu-se usar o valor de p a 60
    x, y = generate_centered_windows(all_modules, 60)
    beta = linear_model(x, y)

    np.sort(outlier_indices)
    x1, y1 = generate_centered_windows_outliers(all_modules, outlier_indices, 60)
    y_prev = linear_model_predict(x1, beta)

    all_modules_no_outliers = removes_outliers_for_predictions(all_modules, outlier_indices, y_prev)

    plot_outlier_replacement_3_11(data, y1, outlier_indices, y_prev)

    return

def ex_4_1():
    acc = calculateModule(np.vstack(sensors_data), 2, 4)
    gyr = calculateModule(np.vstack(sensors_data), 5, 7)
    mag = calculateModule(np.vstack(sensors_data), 8, 10)

    mean_acc = np.mean(acc)
    std_acc = np.std(acc)
    mean_gyr = np.mean(gyr)
    std_gyr = np.std(gyr)
    mean_mag = np.mean(mag)
    std_mag = np.std(mag)

    # normalidade das variaveis para cada atividade
    print("Kolmogorov-Smirnov:")
    r_acc = stats.kstest(acc, 'norm', args=(np.mean(acc), np.std(acc)))
    r_gyr = stats.kstest(gyr, 'norm', args=(np.mean(gyr), np.std(gyr)))
    r_mag = stats.kstest(mag, 'norm', args=(np.mean(mag), np.std(mag)))
    print(r_acc.pvalue, r_gyr.pvalue, r_mag.pvalue)

    print("Shapiro")
    r_acc = stats.shapiro(acc)
    r_gyr = stats.shapiro(gyr)
    r_mag = stats.shapiro(mag)
    print(r_acc.pvalue, r_gyr.pvalue, r_mag.pvalue)

    print("Normal Test")
    r_acc = stats.shapiro(acc)
    r_gyr = stats.shapiro(gyr)
    r_mag = stats.shapiro(mag)
    print(r_acc.pvalue, r_gyr.pvalue, r_mag.pvalue)

    return


#EX4.2 ____________________________________________________________________

def extract_features_by_sensor(data, window_time, overlap):

    # Tamanho da janela e do passo em amostras
    window_size = int(window_time * FS)
    step_size = int(window_size * (1 - overlap))

    # Lista para armazenar as features de cada janela
    feature_list = []

    # Percorrer o sinal criando as janelas
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data[start:end]

        # Extrair features da janela (função fornecida pelo utilizador)
        feats = extract_feature_vector(window)
        feature_list.append(feats)

    # Converter lista em matriz (n_janelas × n_features)
    features_matrix = np.vstack(feature_list)

    print("features shape (4.2):", features_matrix.shape)
    return features_matrix

def main():
    # EX 2
    getFiles(PATH)                  # get all the individuals
    ind = getIndividual(PATH, 0)    # get one individual
    #print(calculateModule(ind[0], 1, 3))

    create_list_by_sensor()

    # EX 3.1

    #num_outliers_per_sensor = boxPlot_modules_3(plot = False)  #This is the right one

    #print(num_outliers_per_sensor)
    
    # EX 3.2 - analyse outliers
    #calculateDensityOutliers(num_outliers_per_sensor)

    # EX 3.3
    # created: z_scores()

    # EX 3.4
    k = 3       # 3 ; 3.5 ; 4

    #ex_3_4(k, plot = False)

    # EX 3.6
    #centroids, labels, distances = kmeans(individuals[0][0][:, 1:4], 16, 100, 1e-4, 40) #Usámos o número de atividades para o número de clusters
    #print(individuals[0][0][:, 1:4].shape)

    
    #print(centroids)
    #print(labels)
    #print(distances)

    # EX 3.7
    #n_outliers, outliers = calculate_outliers_by_centroids(distances, 3, labels)

    #print(n_outliers)

    # EX 3.7
    #graph_3d(centroids, individuals[0][0][:, 1:4], labels, outliers)

    # EX 3.8
    #inject_outliers()

    # EX 3.10
    # create_list_by_sensor()
    # create_list_complete()

    # all_data_in_one_array = np.vstack(people_data)

    # modules_acceleration = calculateModule(all_data_in_one_array, 1, 3)
    # modules_acceleration = modules_acceleration.reshape(-1, 1)

    # ex_3_10(modules_acceleration)

    # EX 3.11

    # modules_gyroscope = calculateModule(all_data_in_one_array, 4, 6)
    # modules_gyroscope = modules_gyroscope.reshape(-1, 1)
    # modules_magnetometer = calculateModule(all_data_in_one_array, 7, 9)
    # modules_magnetometer = modules_magnetometer.reshape(-1, 1)

    # all_modules = np.hstack([modules_acceleration, modules_gyroscope, modules_magnetometer])
    # print(all_modules.shape)

    # ex_3_11(all_modules)

    # EX 4.1
    # ex_4_1()

    # EX 4.2

    all_features_list = []
    for i in range(NUM_SENSORS):
        matrix = extract_features_by_sensor(sensors_data[i], 2, 0.5)
        all_features_list.append(matrix)


    return

if __name__ == "__main__":
    main()