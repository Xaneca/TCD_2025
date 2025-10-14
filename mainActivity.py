import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH = "./../FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master"
activities = pd.read_csv("nameActivities.csv")
titles_sensors = ["pulso esquerdo", "pulso direito", "peito", "perna superior direita", "perna inferior esquerda"]
titles_vectors = ["accelerometer", "gyroscope", "magnetometer"]
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
            plt.title(f"Boxplots de atividades para o modulo do vetor {titles_vectors[i]} do sensor {titles_sensors[k]}")
            bp = plt.boxplot(data_per_x, positions=unique_x_vals, widths=0.6)
            for l, flier in enumerate(bp['fliers']):
                temp = []
                outliers_values = flier.get_ydata()
                temp.append(len(np.unique(outliers_values))) #Coloca o número de outliers sem duplicados
                temp.append(len(data_per_x[l])) #Coloca o número total de valores por cada boxplot para cada atividade
                this_vector_outliers.append(temp)

            plt.xticks(unique_x_vals, [int(x) for x in unique_x_vals]) #Força os valores de x a aparecerem como interiros
            plt.xlabel("Atividade")
            plt.ylabel(f"Valor do modulo do {titles_vectors[i]}")
            plt.grid(True)
            plt.show()
            this_sensor_outliers.append(this_vector_outliers)
        outliers.append(this_sensor_outliers)
    return outliers




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
    plt.figure()
    for s in range(NUM_SENSORS):
        all_x, all_y, all_colors = [], [], []
        for i in range(NUM_PEOPLE):
            module = calculateModule(individuals[i][s], start_idx, start_idx + 2)
            outliers_mask = z_scores(module, k)
            all_x.extend(individuals[i][s][:, -1] + 0.1 * s)
            all_y.extend(module)
            all_colors.extend(np.where(outliers_mask, "red", "blue"))
        plt.scatter(all_x, all_y, c=all_colors, s=10, alpha=0.6, label=f"Sensor {s+1}")
    plt.xticks(range(1, NUM_ACTIVITIES + 1), activities["name"], rotation=90)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    #plt.tight_layout()
    plt.show()

    return

def ex_3_4(k):
    # accelerometter:
    show_outliers(1, k, "Acelerómetro")

    # gyroscope
    show_outliers(4, k, "Giroscópio")

    # mangetometer
    show_outliers(7, k, "Magnetómetro")

    return

def main():
    # EX 2
    getFiles(PATH)                 # get all the individuals
    ind = getIndividual(PATH, 0)    # get one individual
    #print(individuals)
    #print(calculateModule(ind[0], 1, 3))

    # EX 3.1
    #num_outliers_per_activity = boxPlot_modules(plot = True)   # still with outliers
    num_outliers_per_sensor = boxPlot_modules_3(plot = True) 
    #print(num_outliers_per_sensor)
    
    # EX 3.2 - analyse outliers
    calculateDensityOutliers(num_outliers_per_sensor)

    # EX 3.3
    # created: z_scores()

    # EX 3.4
    k = 3       # 3 ; 3.5 ; 4
    ex_3_4(k)

    return

if __name__ == "__main__":
    main()