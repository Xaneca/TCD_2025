import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

PATH = "./../FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master"
PLOT_PATH = "./plots"
activities = pd.read_csv("nameActivities.csv")
titles_sensors = ["Left wrist", "Right wrist", "Chest", "Upper right leg", "Lower left leg"]
titles_vectors = ["Accelerometer", "Gyroscope", "Magnetometer"]
NUM_PEOPLE = 15
NUM_SENSORS = 5
NUM_ACTIVITIES = 16
NUM_COLUNAS = 12
individuals = []    # tam 15 -> tam 5 -> tam 12 + data
                    # (15, 5, 12)
sensors_data = []

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
    # idx - 1,2,3 - accelerometer
    # idx - 4,5,6 - gyroscope
    # idx - 7,8,9 - magnetometer
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
    return sensors_data

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
def kmeans(X, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
    # Configura semente aleatória
    if random_state is not None:
        np.random.seed(random_state)

    # Escolhe aleatoriamente centróides iniciais
    indices = np.random.choice(X.shape[0], n_clusters, replace=False) #Escolhe n_clusters do array X e oreplace = False faz com que não escolha pontos do array X repetidos
    centroids = X[indices] 

    for iteration in range(max_iters):
        # 1️⃣ Atribui cada ponto ao centróide mais próximo
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 2️⃣ Calcula novos centróides
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])

        # 3️⃣ Verifica convergência
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            print(f"Convergência atingida na iteração {iteration + 1}")
            break

        centroids = new_centroids

    return centroids, labels

def ex_3_8(data):
    calculateDensityOutliers(data)    

def ex_4_1():
    distribuiçoes = []
    # normalidade das variaveis para cada atividade
    for s in range(NUM_SENSORS):
        sensor = sensors_data[s]
        sensor[]

    return

def main():
    # EX 2
    getFiles(PATH)                  # get all the individuals
    ind = getIndividual(PATH, 0)    # get one individual
    #print(calculateModule(ind[0], 1, 3))

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

    # EX 3.8
    #ex_3_8()

    # EX 4.1
    
    create_list_by_sensor()
    print(type(sensors_data))


    return

if __name__ == "__main__":
    main()