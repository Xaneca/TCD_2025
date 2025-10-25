import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

PATH = "./../FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master"
activities = pd.read_csv("nameActivities.csv")
titles_sensors = ["Left wrist", "Right wrist", "Chest", "Upper right leg", "Lower left leg"]
titles_vectors = ["Accelerometer", "Gyroscope", "Magnetometer"]
NUM_PEOPLE = 15
NUM_SENSORS = 5
NUM_ACTIVITIES = 16
NUM_COLUNAS = 12
individuals = []    # tam 15 -> tam 5 -> tam 12 + data
                    # (15, 5, 12)

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

    return ind

# def calculateModule(ind, sensor_num, i_x, i_z):
#     # idx - 1,2,3 - accelerometer
#     # idx - 4,5,6 - gyroscope
#     # idx - 7,8,9 - magnetometer
#     return np.sqrt(np.sum(ind[sensor_num][:, i_x:i_z]**2, axis=1))   # axis = 1 -> por linha

def calculateModule(ind_sensor, i_x, i_z):
    # idx - 1,2,3 - accelerometer
    # idx - 4,5,6 - gyroscope
    # idx - 7,8,9 - magnetometer
    return np.sqrt(np.sum(ind_sensor[:, i_x:i_z]**2, axis=1))   # axis = 1 -> por linha

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

        for i in range (3):        
            this_vector_outliers = []

            #Cria os arrays com os valores para y e x
            y_vals = calculateModule(all_the_data, 1 + 3 * i, 3 + 3 * i) #Tira o modulo para um dos vectores
            #print(y_vals)
            x_vals = all_the_data[:, 11] #Retira todos os valores de x
            #print(x_vals)
            unique_x_vals = np.unique(x_vals) #Vai ver os valores unicos das atividades todas - podemos apenas fazer um array de 1 a 12 para poupar tempo -ver Xana

            #print(len(unique_x_vals))
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
            plt.show()
            this_sensor_outliers.append(this_vector_outliers)
        outliers.append(this_sensor_outliers)
    return outliers

def create_list_by_sensor():
    all_by_sensor = []
    for k in range (NUM_SENSORS):
        new_list = []
        for j in range(NUM_PEOPLE):
            new_list.append(individuals[j][k])
        all_the_data = np.vstack(new_list)
        all_by_sensor.append(all_the_data) 
    return all_by_sensor


def calculateDensityOutliers(num_outliers_per_activity):
    print("------- DENSITY ------")
    for s in range(NUM_SENSORS):
        print(f"--------- Sensor {titles_sensors[s]} ---------")
        for v in range(3):
            print(f"  --------- Vector {titles_vectors[v]} ---------")
            for a in range(NUM_ACTIVITIES):
                d = num_outliers_per_activity[s][v][a][0] / num_outliers_per_activity[s][v][a][1] * 100
                print(f"    {activities.loc[a, 'name']}: {d}")
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
            outliers_mask = z_scores(module, k)
            all_x.extend(individuals[i][s][:, -1])
            all_y.extend(module)
            all_colors.extend(np.where(outliers_mask, "red", "blue"))
        plt.figure(figsize =(8, 5))
        plt.scatter(all_x, all_y, c=all_colors, s=10, alpha=0.6)
        plt.xticks(range(1, NUM_ACTIVITIES + 1))
        plt.title(f"{title} for Sensor {titles_sensors[s]}")
        plt.xlabel("Activity")
        plt.ylabel(f"Value of the Vector Module {title}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        #plt.tight_layout()
        plt.show()

    return

def ex_3_4(k):
    # accelerometter:
    show_outliers(1, k, titles_vectors[0])

    # gyroscope
    show_outliers(4, k, titles_vectors[1])

    # mangetometer
    show_outliers(7, k, titles_vectors[2])

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

def calculate_outliers_centroids(distances, k, labels):

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
    current_outliers = np.sum(outlier_mask)
    total = len(data)
    current_density = current_outliers / total * 100

    # Se já há outliers suficientes → termina
    if current_density >= percentage:
        return data, current_outliers

    # Caso contrário, injetar novos outliers
    target_outliers = int(percentage * total)
    n_to_inject = target_outliers - current_outliers

    # Escolher índices aleatórios que não sejam já outliers
    candidate_indices = np.where(~outlier_mask)[0]
    np.random.shuffle(candidate_indices)
    inject_indices = candidate_indices[:n_to_inject]

    # s ∈ {-1, +1}
    s = np.random.choice([-1, 1], size=n_to_inject)

    # q ∈ [0, z)
    q = np.random.uniform(0, z, size=n_to_inject)

    # Aplicar fórmula: μ + s × k × (σ + q)
    new_values = mean + s * k * (std_dev + q)

    # Injetar os novos valores
    data[inject_indices] = new_values

    return data, target_outliers

# EX3.9
def linear_model(X, y):
    # Adicionar a coluna de 1s para o intercepto (β0)
    X = np.column_stack((np.ones(X.shape[0]), X))

    # Calcular betas com a fórmula (XᵀX)⁻¹ Xᵀ y
    beta = np.linalg.pinv(X) @ y
    print(beta)

    return beta

#EX3.9
def linear_model_predict(X, beta):
    #preve o y pelo modelo linear
    y_pred = X @ beta
    print(y_pred)

    return y_pred


def generate_windows(data, p):
    x = []
    y = []
    for i in range (len(data)):
        x.append(data[i:i+p,:])
        y.append(data[p+i])
    final_x = np.vstack(x)
    final_y = np.vstack(y)

    return final_x, final_y



def main():
    # EX 2
    getFiles(PATH)                 # get all the individuals
    ind = getIndividual(PATH, 0)    # get one individual
    #print(individuals)
    #print(calculateModule(ind[0], 1, 3))

    # EX 3.1
    #num_outliers_per_activity = boxPlot_modules(plot = True)   # still with outliers
    #num_outliers_per_sensor = boxPlot_modules_3(plot = True)  #This is the right one
    #print(num_outliers_per_sensor)
    
    # EX 3.2 - analyse outliers
    #calculateDensityOutliers(num_outliers_per_sensor)

    # EX 3.3
    # created: z_scores()

    # EX 3.4
    k = 3       # 3 ; 3.5 ; 4
    #ex_3_4(k)

    # EX 3.6
    #centroids, labels, distances = kmeans(individuals[0][0][:, 1:4], 16, 100, 1e-4, 40) #Usámos o número de atividades para o número de clusters
    #print(individuals[0][0][:, 1:4].shape)

    
    #print(centroids)
    #print(labels)
    #print(distances)

    #EX3.7
    #n_outliers, outliers = calculate_outliers_by_centroids(distances, 3, labels)

    #print(n_outliers)

    #EX3.7
    #graph_3d(centroids, individuals[0][0][:, 1:4], labels, outliers)


    #EX3.10
    all_sensors_list = create_list_by_sensor()

    all_data_in_one_array = np.vstack(all_sensors_list)

    modules_acceleration = calculateModule(all_data_in_one_array, 1, 4)

    x, y = generate_windows(modules_acceleration, 5)

    linear_model(x, y)
    


    return

if __name__ == "__main__":
    main()