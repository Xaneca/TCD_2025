import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from scipy.stats import skew, kurtosis, iqr, pearsonr
from numpy.fft import fft, fftfreq
from features import extract_feature_vector
import os


PATH = "./../../FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master"
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

people_sensor_data = []

def getFiles(path):
    for i in range(NUM_PEOPLE):
        ind = []
        for j in range(NUM_SENSORS):
            df = pd.read_csv(f'{path}/part{i}/part{i}dev{j + 1}.csv')
            ind.append(df.to_numpy())
        individuals.append(ind)
    print("Individuals Extracted")
    return

def getIndividual(path, ind_num):
    ind = []
    for j in range(NUM_SENSORS):
        df = pd.read_csv(f'{path}/part{ind_num}/part{ind_num}dev{j + 1}.csv')
        ind.append(df.to_numpy())
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

def all_modules():
    s = np.vstack(sensors_data)
    modules = []
    acc = np.vstack(np.sqrt(np.sum(s[:, 1:4 + 1]**2, axis=1)))
    gyr = np.vstack(np.sqrt(np.sum(s[:, 4:7 + 1]**2, axis=1)))
    mag = np.vstack(np.sqrt(np.sum(s[:, 7:10 + 1]**2, axis=1)))

    all_mod = np.concatenate((acc, gyr, mag), axis = 1)

    return all_mod

def boxPlot_modules(plot = True, save = False):
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
            if save:
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


def create_list_by_sensor_labelp():
    for k in range(NUM_SENSORS):
        new_list = []
        for j in range(NUM_PEOPLE):
            data = individuals[j][k]              # array original
            person_col = np.full((data.shape[0], 1), j + 1)  # coluna com j+1
            data_with_person = np.hstack((data, person_col))
            new_list.append(data_with_person)

        all_the_data = np.vstack(new_list)
        sensors_data.append(all_the_data)
    
    print(len(sensors_data))
    print(len(sensors_data[0]))
    print(sensors_data[0].shape)


def create_list_by_people_and_sensor():
    #all_by_sensor = []
    for k in range (NUM_PEOPLE):
        new_list = []
        for j in range(NUM_SENSORS):
            new_list.append(individuals[k][j])
        people_sensor_data.append(new_list) 

    print(len(people_sensor_data))
    print(len(people_sensor_data[0]))
    print(len(people_sensor_data[0][0]))
    print(people_sensor_data[0][0:10])


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

def calculateDensityOutliers(num_outliers_per_activity):
    print("------- DENSITY (all sensors) ------")
    for v in range(3):  # 0=ACC,1=GYR,2=MAG
        print(f"--------- Vector {titles_vectors[v]} ---------")
        for a in range(NUM_ACTIVITIES):
            total_numerator = 0
            total_denominator = 0
            for s in range(NUM_SENSORS):
                numerator = num_outliers_per_activity[s][v][a][0]
                denominator = num_outliers_per_activity[s][v][a][1]
                total_numerator += numerator
                total_denominator += denominator

            d = density(total_numerator, total_denominator)
            print(f"    {activities.loc[a, 'name']}: {d:.2f}%")
    return

def z_scores(data, k=None):
    """ outliers_mas k -> destaca os pontos que são outliers
        z -> devolve o cálculo do z-score para normalizaçao  """

    data = np.asarray(data, dtype=float)

    # Calcular média e desvio padrão por coluna (axis=0)
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    if std_dev == 0:
        std_dev = 1e-8  # evitar divisão por zero

    # Normalização vetorizada
    z = (data - mean) / std_dev

    # Máscara de outliers opcional
    outliers_mask = None
    if k is not None:
        outliers_mask = np.abs(z) > k

    return outliers_mask, z

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

def show_outliers(start_idx, k, title, plot = True, save = False):
    # sensors_list = create_list_by_sensor()        # já é corrido no inicio do codigo
    vector_outliers = []

    for s in range(NUM_SENSORS):
        print(f"\tSensor {titles_sensors[s]}")
        colors = []
        this_sensor = sensors_data[s]
        y = calculateModule(this_sensor, start_idx, start_idx + 2)
        x = this_sensor[:,-1]   # activity value
        outliers_mask, _ = z_scores(y, k)

        for act in np.unique(x):
            mask = x == act
            y_act = y[mask]
            mean = np.mean(y_act)
            std = np.std(y_act)

            if std != 0:
                z = (y_act - mean) / std
                outliers_mask[mask] = np.abs(z) > k  # marca só os dessa atividade

        # numero de outliers e numero de pontos totais:
        num_outliers = np.sum(outliers_mask)
        total_points = len(outliers_mask)
        vector_outliers.append([num_outliers, total_points])

        colors.extend(np.where(outliers_mask, "red", "blue"))
        #print(colors[:50])
        plt.figure(figsize = (8,5))
        plt.scatter(x[~outliers_mask], y[~outliers_mask], c="blue", s=10, alpha=0.6, label="Normais")
        plt.scatter(x[outliers_mask], y[outliers_mask], c="red", s=10, alpha=0.6, label="Outliers")
        plt.xticks(range(1, NUM_ACTIVITIES + 1))
        plt.title(f"{title} for Sensor {titles_sensors[s]}")
        plt.xlabel("Activity")
        plt.ylabel(f"Value of the Vector Module {title}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        if save:
            plt.savefig(PLOT_PATH + "/ex3_4_k_3" + f"/sensor{s}_{title}.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace
        if plot:
            plt.show()
        plt.close()

    return vector_outliers

def ex_3_4(k, plot = True, save = False):
    all_outlier_vectors = []

    print("Executing z_score outliers:")

    # accelerometter:
    print(f"Vector {titles_vectors[0]}:")
    all_outlier_vectors.append(show_outliers(1, k, titles_vectors[0], plot, save))

    # gyroscope
    print(f"Vector {titles_vectors[1]}:")
    all_outlier_vectors.append(show_outliers(4, k, titles_vectors[1], plot, save))

    # mangetometer
    print(f"Vector {titles_vectors[2]}:")
    all_outlier_vectors.append(show_outliers(7, k, titles_vectors[2], plot, save))

    return all_outlier_vectors

def plot_z_outlier_heatmap_from_list(results, save = False, plot = False):
    heatmap_data = np.zeros((NUM_SENSORS, 3))  # sensores × vetores

    for v in range(3):
        for s in range(NUM_SENSORS):
            n_out, total = results[v][s]
            heatmap_data[s, v] = density(n_out, total) if total > 0 else 0

    plt.figure(figsize=(8,6))
    im = plt.imshow(heatmap_data, cmap="Reds", aspect="auto")
    plt.xticks(range(3), titles_vectors)
    plt.yticks(range(NUM_SENSORS), titles_sensors)
    plt.title("Densidade Global de Outliers (%)")
    plt.xlabel("Vetor")
    plt.ylabel("Sensor")

    for i in range(NUM_SENSORS):
        for j in range(3):
            plt.text(j, i, f"{heatmap_data[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, label="Densidade de Outliers (%)")
    plt.tight_layout()
    if save:
        plt.savefig(PLOT_PATH + "/ex3_4_k_3" + f"/density_HeatMap.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace
    if plot:
        plt.show()
    plt.close()


# EX 3.6

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

def graph_3d(centroids, data, labels, outliers, vec = "Default", sensor = "6", plot = True):
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

    if plot:
        plt.show()
    plt.savefig(PLOT_PATH + "/ex3_7" + f"/kmeans_{vec}_{sensor}.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace
    plt.close()

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

    return n_outliers, outliers, distances.shape[0] # pontos totais

def matrix_by_sensor(sensor_index, labels):
    number_labels = np.unique(labels[sensor_index])
    number_activities, counts = np.unique(sensors_data[sensor_index][:, -1], return_counts=True)
    counts_dict = dict(zip(number_activities, counts))

    # Cria uma matriz 16x16 inicializada com zeros
    matriz = np.zeros((len(number_activities), len(number_activities)), dtype=int)

    # Preenche a matriz com contagens
    for a, b in zip(labels[sensor_index], sensors_data[sensor_index][:, -1]):
        matriz[a, int(b-1)] += 1   # -1 porque os índices começam em 0

    for i in range(matriz.shape[0]):
        valor = i + 1  # porque as classes começam em 1
        if valor in counts_dict:
            matriz[i, :] = (matriz[i, :] / counts_dict[valor]) * 100 

    # Visualização
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matriz, cmap="Blues")

    # Adiciona rótulos
    ax.set_xticks(np.arange(len(number_activities)))
    ax.set_yticks(np.arange(len(number_activities)))
    ax.set_xticklabels(np.arange(1, len(number_activities) + 1))
    ax.set_yticklabels(np.arange(1, len(number_activities) + 1))

    # Mostra o valor dentro de cada célula
    for i in range(len(number_activities)):
        for j in range(len(number_activities)):
            ax.text(j, i, f"{matriz[i, j]:.1f}%",   # mostra com 1 casa decimal e o símbolo %
                    ha="center", va="center",
                    color="black" if matriz[i, j] < matriz.max()/2 else "white")

    # Títulos e labels
    ax.set_xlabel("Labels da Lista 2 (Preditas)")
    ax.set_ylabel("Labels da Lista 1 (Verdadeiras)")
    ax.set_title("Matriz 16x16 - Contagem de combinações de labels")

    plt.colorbar(im)
    plt.tight_layout()  # adiciona espaço entre elementos
    plt.show()

def ex_3_7(vec, ind_start, plot = True):
    # before
    # all_mod = all_modules()
    # centroids, labels, distances = kmeans(individuals[0][0][:, 1:4], 16, 100, 1e-4, 40) #Usámos o número de atividades para o número de clusters
    # centroids, labels, distances = kmeans(all_mod, 16, 100, 1e-4, 40)
    # print(individuals[0][0][:, 1:4].shape)
    #print(centroids)
    #print(labels)
    #print(distances)

    # EX 3.7
    # n_outliers, outliers = calculate_outliers_by_centroids(distances, 3, labels)

    #print(n_outliers)

    # EX 3.7
    # graph_3d(centroids, individuals[0][0][:, 1:4], labels, outliers)
    # graph_3d(centroids, all_mod, labels, outliers)

    ####################################################3

    list_density = []
    labels_by_sensor = []

    # vetor
    print("Vector: ", vec)
    for i in range(NUM_SENSORS):
        centroids, labels, distances = kmeans(sensors_data[i][:, ind_start:ind_start+3], 16, 100, 1e-4, 40)
        n_outliers, outliers, n_total = calculate_outliers_by_centroids(distances, 3, labels)
        list_density.append(n_outliers/n_total*100)
        labels_by_sensor.append(labels)
        print("sensor", i, n_outliers/n_total*100, "%")
        graph_3d(centroids, sensors_data[i][:, ind_start:ind_start+3], labels, outliers, vec, i, plot)
    
    return list_density, labels_by_sensor


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
    n_to_inject = target_outliers - current_outliers

    # Escolher índices aleatórios que não sejam já outliers
    candidate_indices = np.where(~outlier_mask)[0]
    candidate_indices = candidate_indices[candidate_indices >= 100]
    np.random.shuffle(candidate_indices)
    inject_indices = candidate_indices[:n_to_inject]

    # s ∈ {-1, +1}
    s = np.random.choice([-1, 1], size=n_to_inject)

    # q ∈ [0, z)
    q = np.random.uniform(0, z, size=n_to_inject)

    # Aplicar fórmula: μ + s × k × (σ + q)
    new_values = mean + s * k * (std_dev + q)

    # Injetar os novos valores
    new_values_1d = new_values.ravel()
    data[inject_indices] = new_values_1d.reshape(-1, 1)

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
    """
    Substitui os valores outlier pelos valores previstos pelo modelo.

    data : array 1D (série temporal)
    outlier_indices : índices dos outliers
    y_prev : previsões (n_outliers x 1) ou (n_outliers,)
    """
    # Garantir vetor 1D de escalares
    y_prev = np.asarray(y_prev).reshape(-1)

    for i, index in enumerate(outlier_indices):
        data[index] = y_prev[i]

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

def distribuition_of_prediction_errors(y, y_prev):
    errors = y - y_prev 

    plt.figure(figsize=(8,4))
    plt.hist(errors, bins=30, edgecolor='k')
    plt.xlabel("Erro de predição")
    plt.ylabel("Frequência")
    plt.title("Distribuição do erro de predição")
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
    plt.savefig(PLOT_PATH + "/ex3_10" + f"/crossoverValidation.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace
    plt.show()

    k = 2 #Valor normalmente usado para k
    data, target_outliers, outlier_indices = inject_outliers(modules_acceleration, k, 10, 1)

    # Para esta iteração escolher o p que tiver menos erro no cross_validation depois de rodar o código pelo gráfico e valores do erro médio decidiu-se usar o valor de p a 100
    x, y = generate_windows(modules_acceleration, 100)
    beta = linear_model(x, y)

    np.sort(outlier_indices)
    x1, y1 = generate_windows_outliers(modules_acceleration, outlier_indices, 100)
    y_prev = linear_model_predict(x1, beta)

    distribuition_of_prediction_errors(y1, y_prev)

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

# EX 4.1
def test_normality(acc, gyr, mag):
        # normalidade das variaveis para cada atividade


    # usado para samples pequenas 
    print("Shapiro")
    r_acc = stats.shapiro(acc)
    r_gyr = stats.shapiro(gyr)
    r_mag = stats.shapiro(mag)
    print(r_acc.pvalue, r_gyr.pvalue, r_mag.pvalue)

    print("Normal Test")
    r_acc = stats.normaltest(acc)
    r_gyr = stats.normaltest(gyr)
    r_mag = stats.normaltest(mag)
    print(r_acc.pvalue, r_gyr.pvalue, r_mag.pvalue)

    print("Kolmogorov-Smirnov:")
    r_acc = stats.kstest(acc, 'norm', args=(np.mean(acc), np.std(acc)))
    r_gyr = stats.kstest(gyr, 'norm', args=(np.mean(gyr), np.std(gyr)))
    r_mag = stats.kstest(mag, 'norm', args=(np.mean(mag), np.std(mag)))
    print(r_acc.pvalue, r_gyr.pvalue, r_mag.pvalue)

    return r_acc.pvalue, r_gyr.pvalue, r_mag.pvalue     # retorn p de kolmogorov-smirnov 

def activities_ids():
    new_list = []
    for k in range (NUM_SENSORS):
        for j in range(NUM_PEOPLE):
            new_list.extend(individuals[j][k][:,-1])
    return new_list

def statisticalTest(module):

    act_id = activities_ids()
    # print(act_id[0])
    # print(module[0])

    atividades = []

    act_id = np.array(activities_ids())
    module = np.array(module)

    # print(len(act_id))
    # print(len(module)) # valor 3664563

    for i in range(1, NUM_ACTIVITIES + 1):
        #print("atividade", i)
        atividade = module[act_id == i]
        atividades.append(atividade)

    # for i, a in enumerate(atividades):
    #     print(f"Atividade {i + 1}: tamanho={len(a)}, média={np.mean(a) if len(a) > 0 else 'sem dados'}")


    # H, p = stats.kruskal(atividades[0], atividades[1], atividades[2], atividades[3],
    #                      atividades[4], atividades[5], atividades[6], atividades[7],
    #                      atividades[8], atividades[9], atividades[10], atividades[11],
    #                      atividades[12], atividades[13], atividades[14], atividades[15])

    H, p = stats.kruskal(*atividades)

    print(H, p)

    return

def statisticalTest_OnePerson(acc):
    return

def ex_4_1():
    print("\n\tEXERCICIO 4.1\n")
    acc = calculateModule(np.vstack(sensors_data), 1, 3)
    gyr = calculateModule(np.vstack(sensors_data), 4, 6)
    mag = calculateModule(np.vstack(sensors_data), 7, 9)

    #print(len(acc))

    p_acc, p_gyr, p_mag = test_normality(acc, gyr, mag)

    p_values = [p_acc, p_gyr, p_mag]
    modules = [acc, gyr, mag]
    nomes = ["Acelerometro", "Giroscopio", "Magnetometro"]

    # DADOS UNPAIRED porque juntamos todas as pessoas para avaliar as atividades
    # se quiser comparar atividade com atividade sao 136 valores
    # fisher e chi-square e McNemar sao para categoricos ❌
    # Student's - distribuiçao normal ❌
    # Analysis of variance - unpaired, normal ❌
    # Wilcoxon’s - ordinal/continuous, nao precisa ser normal, paired e unpaired, 2 grupos ✅
    # Kruskal-Wallis - tal como wilcoxon, para unpaired, mais q 2 grupos
    # friedman - nao normal, +2 grupos, paired (varias atividades da mesma pessoa)
    # ...

    
    for i in range(3):
        print("\nKruskal-Wallis")
        if p_values[i] < 0.05:
            print(f"{nomes[i]} (H, p):")
            statisticalTest(modules[i])
        else:
            # NESTE CASO CONCLUIMOS QUE OS PVALUES SAÕ < 0.05 -> logo não têm distribuiçao normal
                # não precisamos fazer codigo para o caso de ter distribuiçao normal
            pass
    
    #statisticalTest_OnePerson(acc)

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

def save_to_file(obj, filename="data.npy", description="Dados"):
    """
    Guarda uma lista ou array em ficheiro .npy usando NumPy.
    
    Parâmetros:
        obj : list ou np.ndarray
            Estrutura de dados a guardar (ex: features ou PCA).
        filename : str
            Nome do ficheiro de saída.
        description : str
            Texto para imprimir no log.
    """
    np.save(filename, np.array(obj, dtype=object))
    print(f"[OK] {description} guardados em '{filename}' ({len(obj)} sensores).")

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

def ex_4_2_ancient():
    print("=== Verificando ficheiro de features ===")

    filename = "all_features_norm.npy"

    # Tenta carregar primeiro
    loaded_features = load_from_file(filename)

    if loaded_features is not None:
        print("[OK] Features já existentes foram carregadas.")
        all_features_list_norm = loaded_features

    else:
        print("[INFO] Ficheiro não encontrado. Criando features...")

        all_features_list = []
        for i in range(NUM_SENSORS):
            print(f"\tSensor: {i}")
            matrix = extract_features_by_sensor(sensors_data[i], 2, 0.5)
            all_features_list.append(matrix)

        print("\nExemplo (primeiras 20 linhas do sensor 0):")
        print(all_features_list[0][:20])

        print("\nNormalizando:")
        all_features_list_norm = []
        for i in range(NUM_SENSORS):
            print(f"\tSensor: {i}")
            _, z = z_scores(all_features_list[i])
            all_features_list_norm.append(z)

        # Guardar no ficheiro
        save_to_file(all_features_list_norm, filename)
        print(f"[OK] Features criadas e guardadas em '{filename}'.")

    print("\nShape do sensor 0:", all_features_list_norm[0].shape)
    # print("Primeiras 20 linhas normalizadas do sensor 0:")
    # print(all_features_list_norm[0][:20])
    return all_features_list_norm

def ex_4_2():
    print("=== Verificando ficheiro de features ===")

    filename = "all_features_norm.npy"

    # Tenta carregar primeiro
    loaded_features = load_from_file(filename)

    if loaded_features is not None:
        print("[OK] Features já existentes foram carregadas.")
        all_features_list_norm = loaded_features

    else:
        print("[INFO] Ficheiro não encontrado. Criando features...")

        all_features_list = []

        for i in range(NUM_SENSORS):
            print(f"\nSensor {i}:")
            sensor_data = sensors_data[i]
            X = sensor_data[:, :-1]     # dados do sensor
            labels = sensor_data[:, -1] # labels (1–16)

            sensor_features_by_activity = []

            for act in range(1, NUM_ACTIVITIES + 1):
                print(f"\tAtividade {act}...")

                # Selecionar apenas as amostras desta atividade
                act_data = X[labels == act]

                # Extrair features por janelas (2 s, 50% overlap)
                matrix = extract_features_by_sensor(act_data, 2, 0.5)

                sensor_features_by_activity.append(matrix)

            all_features_list.append(sensor_features_by_activity)

        # Converter tudo em numpy (estrutura 5 × 16, com matrizes internas)
        all_features_list = np.array(all_features_list, dtype=object)

        print("\nNormalizando:")
        all_features_list_norm = []

        for i in range(NUM_SENSORS):
            print(f"\tSensor: {i}")
            norm_sensor_features = []

            for act in range(NUM_ACTIVITIES):
                _, z = z_scores(all_features_list[i][act])
                norm_sensor_features.append(z)

            all_features_list_norm.append(norm_sensor_features)

        all_features_list_norm = np.array(all_features_list_norm, dtype=object)

        # Guardar no ficheiro
        save_to_file(all_features_list_norm, filename)
        print(f"[OK] Features criadas e guardadas em '{filename}'.")

    # Exemplo de confirmação
    print("\nExemplo de shape:")
    print("Sensor 0, Atividade 0 ->", all_features_list_norm[0][0].shape)

    return all_features_list_norm

def ex_4_2_person_sensor():
    print("=== Verificando ficheiro de features ===")

    filename = "all_features_norm_person_sensor.npy"

    # Tenta carregar primeiro
    loaded_features = load_from_file(filename)

    if loaded_features is not None:
        print("[OK] Features já existentes foram carregadas.")
        all_features_list_norm = loaded_features

    else:
        print("[INFO] Ficheiro não encontrado. Criando features...")

        all_features_list = []

        for i in range(NUM_PEOPLE):
            print(f"\nPerson {i}:")
            person_data = people_sensor_data[i]
            features_by_sensor = []
            for l in range(NUM_SENSORS):
                print(f"\nSensor {l}:")
                sensor_data = person_data[l]
                X = sensor_data[:, :-1]     # dados do sensor
                labels = sensor_data[:, -1] # labels (1–16)

                sensor_features_by_activity = []

                for act in range(1, NUM_ACTIVITIES + 1):
                    print(f"\tAtividade {act}...")

                    # Selecionar apenas as amostras desta atividade
                    act_data = X[labels == act]

                    # Extrair features por janelas (2 s, 50% overlap)
                    matrix = extract_features_by_sensor(act_data, 2, 0.5)

                    sensor_features_by_activity.append(matrix)

                features_by_sensor.append(sensor_features_by_activity)

            all_features_list.append(features_by_sensor)

        # Converter tudo em numpy (estrutura 5 × 16, com matrizes internas)
        all_features_list = np.array(all_features_list, dtype=object)

        # print("\nNormalizando:")
        # all_features_list_norm = []

        # for i in range(NUM_PEOPLE):
        #     print(f"\tPerson: {i}")
        #     norm_sensor_features = []
        #     for l in range(NUM_SENSORS):
        #         for act in range(NUM_ACTIVITIES):
        #             _, z = z_scores(all_features_list[i][l][act])
        #             norm_sensor_features.append(z)

        #         all_features_list_norm.append(norm_sensor_features)

        all_features_list_norm = np.array(all_features_list, dtype=object)

        # Guardar no ficheiro
        save_to_file(all_features_list_norm, filename)
        print(f"[OK] Features criadas e guardadas em '{filename}'.")

    # Exemplo de confirmação
    print("\nExemplo de shape:")
    print("Pessoa 0, Sensor 0, Atividade 0 ->", all_features_list_norm[0][0].shape)

    return all_features_list_norm

def ex_4_2_person_sensor_2():
    print("=== Verificando ficheiro de features ===")

    filename = "all_features_norm_person_sensor.npy"
    log_filename = "shapes_person_sensor_activity.txt"

    # Tenta carregar primeiro
    loaded_features = load_from_file(filename)

    if loaded_features is not None:
        print("[OK] Features já existentes foram carregadas.")
        all_features_list_norm = loaded_features

    else:
        print("[INFO] Ficheiro não encontrado. Criando features...")

        all_features_list = []

        with open(log_filename, "w", encoding="utf-8") as log:
            log.write("Pessoa | Sensor | Atividade | Shape Features\n")
            log.write("-" * 50 + "\n")

            for i in range(NUM_PEOPLE):
                print(f"\nPerson {i}:")
                person_data = people_sensor_data[i]
                features_by_sensor = []

                for l in range(NUM_SENSORS):
                    print(f"  Sensor {l}:")
                    sensor_data = person_data[l]
                    X = sensor_data[:, :-1]     # dados do sensor
                    labels = sensor_data[:, -1] # labels (1–16)

                    sensor_features_by_activity = []

                    for act in range(1, NUM_ACTIVITIES + 1):
                        print(f"\tAtividade {act}...")

                        # Selecionar apenas as amostras desta atividade
                        act_data = X[labels == act]

                        # Extrair features por janelas
                        matrix = extract_features_by_sensor(act_data, 2, 0.5)

                        # ====== LOG DA SHAPE ======
                        log.write(
                            f"{i+1:6d} | {l+1:6d} | {act:9d} | {matrix.shape}\n"
                        )

                        sensor_features_by_activity.append(matrix)

                    features_by_sensor.append(sensor_features_by_activity)

                all_features_list.append(features_by_sensor)

        # Converter para numpy object
        all_features_list_norm = np.array(all_features_list, dtype=object)

        # Guardar no ficheiro
        save_to_file(all_features_list_norm, filename)
        print(f"[OK] Features criadas e guardadas em '{filename}'.")
        print(f"[OK] Log de shapes guardado em '{log_filename}'.")

    # Exemplo de confirmação no ecrã
    print("\nExemplo de shape:")
    print("Pessoa 0, Sensor 0, Atividade 0 ->",
          all_features_list_norm[0][0][0].shape)

    return all_features_list_norm

def ex_4_2_person_sensor_final():
    print("=== Verificando ficheiro de features ===")

    filename = "all_features_norm_person_sensor.npy"
    log_filename = "shapes_person_activity_sensors.txt"

    loaded_features = load_from_file(filename)

    if loaded_features is not None:
        print("[OK] Features já existentes foram carregadas.")
        all_features_list_norm = loaded_features

    else:
        print("[INFO] Ficheiro não encontrado. Criando features...")

        all_features_list = []

        with open(log_filename, "w", encoding="utf-8") as log:
            # Cabeçalho
            log.write(
                "Pessoa | Atividade | Sensor1 | Sensor2 | Sensor3 | Sensor4 | Sensor5\n"
            )
            log.write("-" * 90 + "\n")

            for i in range(NUM_PEOPLE):
                print(f"\nPerson {i}:")
                person_data = people_sensor_data[i]
                features_by_sensor = []

                for l in range(NUM_SENSORS):
                    print(f"  Sensor {l}:")
                    sensor_data = person_data[l]
                    X = sensor_data[:, :-1]
                    labels = sensor_data[:, -1]

                    sensor_features_by_activity = []

                    for act in range(1, NUM_ACTIVITIES + 1):
                        act_data = X[labels == act]
                        matrix = extract_features_by_sensor(act_data, 2, 0.5)
                        sensor_features_by_activity.append(matrix)

                    features_by_sensor.append(sensor_features_by_activity)

                # ====== LOG POR PESSOA + ATIVIDADE ======
                for act in range(NUM_ACTIVITIES):
                    shapes = []
                    for sensor in range(NUM_SENSORS):
                        shapes.append(features_by_sensor[sensor][act].shape)

                    log.write(
                        f"{i+1:6d} | {act+1:9d} | "
                        + " | ".join([str(s) for s in shapes])
                        + "\n"
                    )

                all_features_list.append(features_by_sensor)

        all_features_list_norm = np.array(all_features_list, dtype=object)

        save_to_file(all_features_list_norm, filename)
        print(f"[OK] Features criadas e guardadas em '{filename}'.")
        print(f"[OK] Log de shapes guardado em '{log_filename}'.")

    # Confirmação no ecrã
    print("\nExemplo de shape:")
    print("Pessoa 0, Sensor 0, Atividade 0 ->",
          all_features_list_norm[0][0][0].shape)

    return all_features_list_norm


# EX 4.3

def PCA(X, name_file, num_sensor, num_act, plot = True):
    # First center data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Matriz covariancia - mede a variação conjunta entre as features
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Autovalores e autovetores
    # Autovetores → direções principais (componentes)
    # Autovalores → variância explicada em cada direção
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # Ordenar por variância decrescente
    idx = np.argsort(eig_vals)[::-1] 
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Calcular variância explicada (%)
    explained_variance_ratio = eig_vals / np.sum(eig_vals)
    print("Variância explicada por componente:")
    var_acumulada = np.cumsum(explained_variance_ratio)

    num_comp = np.argmax(var_acumulada >= 0.75) + 1
    print(f"Para explicar 75% da variância, são necessárias {num_comp} features.")

    # Projetar os dados nos componentes principais
    # (isto gera os novos eixos principais)
    X_pca = np.dot(X_centered, eig_vecs[:, :num_comp])

    # exemplo = X_pca[0]  # por ex., a primeira janela
    # print(f"Features comprimidas (instante 0):\n{exemplo}")

    # Gráfico 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, color='teal')
    plt.title("Projeção PCA (Componentes 1 e 2)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True)
    plt.savefig(PLOT_PATH + "/ex4_3" + f"/{name_file}.png", dpi=300, bbox_inches="tight")  
    if plot:
        plt.show()
    plt.close()

    # variancia
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(var_acumulada) + 1), var_acumulada, marker='o')
    plt.axhline(0.75, color='r', linestyle='--', label='75% Variância')
    plt.axvline(num_comp, color='g', linestyle='--', label=f'{num_comp} componentes')
    plt.title(f"PCA Sensor {num_sensor}, Activity {num_act} — Variância Explicada Acumulada")
    plt.xlabel("Número de Componentes Principais")
    plt.ylabel("Variância Explicada Acumulada")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f"/ex4_3/{name_file}.png", dpi=300, bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    X_pca_75 = np.dot(X, eig_vecs[:, :num_comp])
    i = 0  # primeira amostra / janela
    print("Features comprimidas (instante 0):", X_pca_75[i])

    return X_pca_75

def run_PCA_by_activity(sensor, num_sensor, PLOT_PATH, plot = True):
    os.makedirs(f"{PLOT_PATH}/ex4_3", exist_ok=True)

    print(f"\n=== Iniciando PCA para Sensor {num_sensor} ===")

    pca_75_list = []

    for act_idx, X in enumerate(sensor, start=1):
        if X is None or len(X) == 0:
            print(f"[AVISO] Sensor {num_sensor}, atividade {act_idx}: sem dados.")
            continue

        print(f"\n--- Sensor {num_sensor} | Atividade {act_idx} ---")
        x_pca_75 = PCA(X, f"pca_{num_sensor}_act{act_idx}", num_sensor, act_idx, plot)  # chamada da tua função original
        pca_75_list.append(x_pca_75)

    print(f"\n[OK] PCA concluído para todas as atividades do sensor {num_sensor}.")

    return pca_75_list

def plot_global_pca_clusters(pca_75_list, num_activities=16):
    """
    Mostra num único gráfico 2D as projeções PCA (75%) de todos os sensores e atividades.
    Cada cor = atividade, cada marcador = sensor.
    """
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, num_activities))
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']  # diferentes marcadores por sensor

    for s_idx, sensor_data in enumerate(pca_75_list):
        for act_idx, X_pca in enumerate(sensor_data, start=1):
            X_pca = np.array(X_pca)
            
            if X_pca is None or len(X_pca) == 0:
                continue
            plt.scatter(X_pca[:, 0], X_pca[:, 1],
                        s=20, alpha=0.6,
                        c=[colors[act_idx-1]],
                        marker=markers[s_idx % len(markers)],
                        label=f"S{s_idx}-A{act_idx}")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=3)

    plt.title("Projeção PCA Global — Sensores e Atividades")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH + f"/ex4_3/heatMap_pca_75.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Tenta carregar primeiro
    loaded_features = load_features_from_file(filename)

    if loaded_features is not None:
        print("[OK] Features já existentes foram carregadas.")
        all_features_list_norm = loaded_features

    else:
        print("[INFO] Ficheiro não encontrado. Criando features...")

        all_features_list = []
        for i in range(NUM_SENSORS):
            print(f"\tSensor: {i}")
            matrix = extract_features_by_sensor(sensors_data[i], 2, 0.5)
            all_features_list.append(matrix)

        print("\nExemplo (primeiras 20 linhas do sensor 0):")
        print(all_features_list[0][:20])

        print("\nNormalizando:")
        all_features_list_norm = []
        for i in range(NUM_SENSORS):
            print(f"\tSensor: {i}")
            _, z = z_scores(all_features_list[i])
            all_features_list_norm.append(z)

        # Guardar no ficheiro
        save_features_to_file(all_features_list_norm, filename)
        print(f"[OK] Features criadas e guardadas em '{filename}'.")

    print("\nShape do sensor 0:", all_features_list_norm[0].shape)
    # print("Primeiras 20 linhas normalizadas do sensor 0:")
    # print(all_features_list_norm[0][:20])
    return all_features_list_norm


#EX4.5----------------------------------------------------------------------------------

def act_fisher_score(X, mu_f):
    n_samples, n_features = X.shape
    mu_fc = np.mean(X, axis=0)
    sigma_fc2 = np.var(X, axis=0, ddof=1)

    numerador = n_samples * (mu_fc - mu_f)**2
    denominador = n_samples * sigma_fc2

    return numerador, denominador

def fisher_score(all_features_norm):
    print(all_features_norm[0][0].shape)
    num_feat = all_features_norm[0][0].shape[1]
    all_scores = np.zeros((NUM_SENSORS, num_feat)) # (5, 110)

    for s in range(NUM_SENSORS):
        print(f"Sensor {s}:")
        sensor_score = np.zeros(num_feat)

        # TODO: calcular media de todos
        mu_f = np.mean(np.vstack(all_features_norm[s]), axis=0)  # valor errado

        numerador = np.zeros((NUM_ACTIVITIES, num_feat))
        denominador = np.zeros((NUM_ACTIVITIES, num_feat))
        for a in range(NUM_ACTIVITIES):
            # print(f"activity {a}:")
            numerador[a], denominador[a] = act_fisher_score(all_features_norm[s][a], mu_f)
            # print(f"numerador: {numerador[a]}\ndenominador: {denominador[a]}\n")

        sensor_score = np.sum(numerador, axis=0) / (np.sum(denominador, axis=0) + 1e-8)
        
        all_scores[s] = sensor_score

    return all_scores

def prepares_data_for_reliefF(all_data_norm): 
    X_sensors = []  # lista para armazenar os dados de cada sensor
    y_sensors = []  # lista para armazenar os labels de cada sensor

    for big_list in all_data_norm:  # percorre os 5 sensores
        X_sensor = []
        y_sensor = []
        
        for class_idx, arr in enumerate(big_list):  # percorre as 16 classes
            X_sensor.append(arr)  # adiciona o array de amostras da classe
            y_sensor.append(np.full(arr.shape[0], class_idx))  # cria vetor de labels
        
        # Junta todas as classes do sensor num único array
        X_sensor = np.vstack(X_sensor)
        y_sensor = np.hstack(y_sensor)
        
        # Adiciona à lista de sensores
        X_sensors.append(X_sensor)
        y_sensors.append(y_sensor)

    return X_sensors, y_sensors


def reliefF(X, y, k=10):
    """
    ReliefF para seleção de features.
    
    X: array (n_samples, n_features)
    y: array (n_samples,)
    k: número de vizinhos mais próximos
    
    Retorna:
    scores: array (n_features,) - relevância de cada feature
    """
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    
    # Distâncias entre todas as amostras
    D = pairwise_distances(X, metric='euclidean')
    
    for i in range(n_samples):
        Xi = X[i]
        yi = y[i]
        
        # índices dos vizinhos (excluindo a própria amostra)
        sorted_idx = np.argsort(D[i])
        sorted_idx = sorted_idx[sorted_idx != i]
        
        # vizinhos da mesma classe (hits) e de outras classes (misses)
        hits = [j for j in sorted_idx if y[j] == yi][:k]
        misses = [j for j in sorted_idx if y[j] != yi][:k]
        
        # atualizar score para cada feature
        for f in range(n_features):
            diff_hit = np.mean([abs(Xi[f] - X[j, f]) for j in hits])
            diff_miss = np.mean([abs(Xi[f] - X[j, f]) for j in misses])
            
            scores[f] += diff_miss - diff_hit
    
    # normalizar scores
    scores /= n_samples
    return scores

def ex_4_5(all_features_norm):
    # all_scores = fisher_score(all_features_norm)
    # print(all_scores)

    # Supondo que tens os dados normalizados em all_data_norm
    X_sensors, y_sensors = prepares_data_for_reliefF(all_features_norm)

    scores_sensors = []

    # Itera sobre os 5 sensores
    for X_sensor, y_sensor in zip(X_sensors, y_sensors):
        scores = reliefF(X_sensor, y_sensor, k=10)
        scores_sensors.append(scores)

    # Exemplo: mostrar os scores do sensor 0
    print("Scores sensor 0:", scores_sensors[0])

    # Se quiseres ver os top 10 features de cada sensor
    for i, scores in enumerate(scores_sensors):
        top_features = np.argsort(scores)[::-1][:10]
        print(f"Top 10 features sensor {i}:", top_features)

    return

def replace_outliers_linear_model(
    data_list,
    cols_to_check=range(1, 7),
    p=100,
    z_thresh=3
):
    """
    data_list : lista com 5 arrays (N x 13)
    cols_to_check : colunas onde detetar outliers
    p : tamanho da janela temporal
    z_thresh : limiar do z-score
    """

    cleaned_data_list = []

    for sensor_idx, data in enumerate(data_list):
        print(f"\nSensor {sensor_idx}")
        data_clean = data.copy()

        for col in cols_to_check:
            print(f"  Coluna {col}")

            column_data = data_clean[:, col]

            # -------------------------------
            # 1. DETEÇÃO DE OUTLIERS (z-score)
            # -------------------------------
            mean = np.mean(column_data)
            std = np.std(column_data)

            if std == 0:
                print("    Std = 0 → ignorada")
                continue

            z_scores = np.abs((column_data - mean) / std)
            outlier_indices = np.where(z_scores > z_thresh)[0]

            # Garantir que há dados suficientes para janelas
            outlier_indices = outlier_indices[outlier_indices >= p]

            print(f"    Outliers detetados: {len(outlier_indices)}")

            if len(outlier_indices) == 0:
                continue

            # ---------------------------------------
            # 2. TREINAR MODELO COM DADOS NORMAIS
            # ---------------------------------------
            x, y = generate_windows(column_data, p)
            beta = linear_model(x, y)

            # ---------------------------------------
            # 3. PREVER OUTLIERS
            # ---------------------------------------
            x_out, y_true = generate_windows_outliers(
                column_data, outlier_indices, p
            )

            y_prev = linear_model_predict(x_out, beta)

            # ---------------------------------------
            # 4. SUBSTITUIR OUTLIERS
            # ---------------------------------------
            column_data_no_outliers = removes_outliers_for_predictions(
                column_data.copy(),
                outlier_indices,
                y_prev
            )

            data_clean[:, col] = column_data_no_outliers

        cleaned_data_list.append(data_clean)

    return cleaned_data_list

def get_feature_names_one_sensor(sensor_id):
    acc = ["Ax", "Ay", "Az"]
    gyro = ["Gx", "Gy", "Gz"]
    all_channels = acc + gyro

    feature_names = []

    # A. Estatísticas temporais
    temporal_feats = [
        "mean", "median", "std", "variance", "rms",
        "avg_derivative", "skewness", "kurtosis",
        "iqr", "zcr", "mcr"
    ]

    for ch in all_channels:
        for feat in temporal_feats:
            feature_names.append(f"{feat}_{ch}_sensor_{sensor_id}")

    # B. Estatísticas espectrais
    spectral_feats = [
        "dominant_frequency", "energy", "spectral_entropy"
    ]

    for ch in all_channels:
        for feat in spectral_feats:
            feature_names.append(f"{feat}_{ch}_sensor_{sensor_id}")

    # C. Correlações
    correlations = [
        ("Ax", "Ay"), ("Ax", "Az"), ("Ay", "Az"),
        ("Gx", "Gy"), ("Gx", "Gz"), ("Gy", "Gz"),
        ("Ax", "Gx"), ("Ay", "Gy"), ("Az", "Gz"),
        ("Ax", "Gy"), ("Ax", "Gz"), ("Ay", "Gx"),
        ("Ay", "Gz"), ("Az", "Gx"), ("Az", "Gy")
    ]

    for ch1, ch2 in correlations:
        feature_names.append(
            f"corr_{ch1}_{ch2}_sensor_{sensor_id}"
        )

    # D. Features físicas
    physical_feats = [
        "AI", "VI", "SMA",
        "EVA1", "EVA2",
        "CAGH", "AVH", "AVG", "ARATG",
        "AAE", "ARE"
    ]

    for feat in physical_feats:
        feature_names.append(f"{feat}_sensor_{sensor_id}")

    return feature_names


def build_feature_dataset(
    data_with_no_outlires,
    window_time,
    overlap,
    output_csv
):
    """
    data_with_no_outlires : lista com 5 arrays (N x 13)
    window_time : duração da janela (s)
    overlap : overlap [0,1[
    output_csv : path do CSV final
    """

    num_sensors = len(data_with_no_outlires)

    # Referência para labels
    ref = data_with_no_outlires[0]
    activities = ref[:, 11]
    people = ref[:, 12]

    unique_people = np.unique(people)
    unique_activities = np.unique(activities)

    rows = []

    for person in unique_people:
        for activity in unique_activities:

            sensor_features = []
            min_windows = np.inf

            # FEATURES POR SENSOR
            for s in range(num_sensors):
                sensor_data = data_with_no_outlires[s]

                mask = (
                    (sensor_data[:, 11] == activity) &
                    (sensor_data[:, 12] == person)
                )

                # Apenas colunas de sinal (até timestamp inclusive)
                signal_data = sensor_data[mask, :10]

                if signal_data.shape[0] == 0:
                    feats = np.empty((0, 110))
                else:
                    feats = extract_features_by_sensor(
                        signal_data,
                        window_time,
                        overlap
                    )

                sensor_features.append(feats)
                min_windows = min(min_windows, feats.shape[0])

            # Se algum sensor não tiver janelas suficientes → ignora
            if min_windows == 0 or min_windows == np.inf:
                continue

            # ALINHAMENTO (TRUNCAMENTO)
            sensor_features = [
                feats[:min_windows, :]
                for feats in sensor_features
            ]

            # CONCATENAÇÃO (550 features)
            X = np.hstack(sensor_features)  # (min_windows x 550)

            # ADICIONAR LABELS
            activity_col = np.full((min_windows, 1), activity)
            person_col = np.full((min_windows, 1), person)

            X = np.hstack([X, activity_col, person_col])

            rows.append(X)

    # DATASET FINAL
    final_matrix = np.vstack(rows)

    columns = []

    for sensor_id in range(1, num_sensors + 1):
        columns.extend(get_feature_names_one_sensor(sensor_id))

    columns.extend(["activity", "person"])

    df = pd.DataFrame(final_matrix, columns=columns)
    df.to_csv(output_csv, index=False)

    print("Dataset final criado")
    print("Shape:", df.shape)


def main():
    # EX 2
    getFiles(PATH)                   # get all the individuals
    #ind = getIndividual(PATH, 0)    # get one individual
    #print(calculateModule(ind[0], 1, 3))

    #create_list_by_sensor()

    #create_list_by_people_and_sensor()

    create_list_by_sensor_labelp()

    data_with_no_outlires = replace_outliers_linear_model(sensors_data)

    print(len(data_with_no_outlires))
    print(len(data_with_no_outlires[0]))
    print(data_with_no_outlires[0].shape)

    #Defenir a window_time, o overlap e o nome do ficheiro 

    window_time = 2
    overlap = 0.5
    output_csv = "data_prepared_for_training"

    build_feature_dataset(data_with_no_outlires, window_time, overlap, output_csv)

    # EX 3.1
    #num_outliers_per_sensor = boxPlot_modules(plot = False, save = False)  #This is the right one

    #print(num_outliers_per_sensor)
    
    # EX 3.2 - analyse outliers
    #calculateDensityOutliers(num_outliers_per_sensor)

    # EX 3.3
    # created: z_scores()

    # EX 3.4

    # k = 3       # 3 ; 3.5 ; 4

    # results = ex_3_4(k, plot = False, save = True)
    # plot_z_outlier_heatmap_from_list(results, save=True, plot=True)

    # EX 3.6 e 3.7 ----------------------------------
    #list_density_1, labels_by_sensor1 = ex_3_7("Accelerometer", 1, False)
    #Faz para o sensor 0 apenas ou seja Pulso esquerdo
    #matrix_by_sensor(0, labels_by_sensor1) # Imprime uma tabela que compara a posição dos pontos nos clusters efetuados pelo kmeans e a distribuição real desses pontos

    #list_density_2, labels_by_sensor2 = ex_3_7("Gyroscope", 4, False)
    #list_density_3, labels_by_sensor3 = ex_3_7("Magnetometer", 7, False)

    # calculado para n demorar tanto tempo
    # list_density_1 = [1.1137663331830852, 1.2115522117351214, 1.2957499872220466, 1.2967960035310169, 1.7952385509158766]
    # list_density_2 = [1.9545680032030541, 1.8585149531789815, 1.3943040471381971, 2.354220166920793, 1.973024834678902]
    # list_density_3 = [1.025002455171094, 0.7779413589287382, 0.7235597936131203, 0.7327862932348929, 1.1501966729286388]

    # heatmap_data = np.array([list_density_1, list_density_2, list_density_3]).T

    # plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    # plt.colorbar(label='Density')
    # plt.title('Outlier Density Heatmap')
    # plt.xlabel('Vector')
    # plt.ylabel('Sensor')

    # # opcional: mostrar rótulos nos eixos
    # plt.xticks(ticks=range(3), labels=['Accelerometer', 'Gyroscope', 'Magnetometer'])
    # plt.yticks(ticks=range(5), labels=[f'Sensor {i+1}' for i in range(5)])
    # plt.savefig(PLOT_PATH + "/ex3_7" + f"/kmeans_heatmap.png", dpi=300, bbox_inches="tight")  # png, 300dpi, remove extra whitespace
    # plt.show()
    # -----------------------------------------------

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

    #all_features_list_norm = ex_4_2()
    #all_features_list = ex_4_2_person_sensor_final()

    # print(all_features_list_norm[4][14][:,1])
    # print(all_features_list_norm[4][13][:,1])

    # # EX 4.3 - PCA

    # pca_filename = "pca_75_list.npy"
    # pca_75_list = load_from_file(pca_filename)

    # num = 0
    # for i in range(5):
    #     for act in range(16):
    #         num += len(all_features_list_norm[i][act])
    # print("asxfghnjmk",num)

    # if pca_75_list is None:
    #     pca_75_list = []
    #     for i in range(NUM_SENSORS):
    #         x_pca_75 = run_PCA_by_activity(all_features_list_norm[i], i, PLOT_PATH, False)
    #         pca_75_list.append(pca_75_list)
    #     save_to_file(pca_75_list, "pca_75_list.npy", "Resultados PCA")
    # plot_global_pca_clusters(pca_75_list)

    # EX 4.5
    #scores_fs = fisher_score(all_features_list_norm)

    # # Ordenar features
    #for l in range (NUM_SENSORS):
        #idx_fs = np.argsort(scores_fs[l])[::-1]
        #print("Top 10 features por Fisher Score:", idx_fs[:10])



    # scores_relief = reliefF(all_features_list_norm[0], sensors_data[0][:, -1], k=10)
    # idx_relief = np.argsort(scores_relief)[::-1]
    # print("Top 10 features por ReliefF:", idx_relief[:10])

    # ex_4_5(all_features_list_norm)

    return

if __name__ == "__main__":
    main()