import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH = "./../../FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master"
activities = pd.read_csv("nameActivities.csv")
NUM_PEOPLE = 15
NUM_SENSORS = 5
NUM_ACTIVITIES = 12
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

        plt.title(f"{activities.loc[activity - 1, "name"]}")
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
        outliers.append({
            "num_outliers": this_activity_outliers / 3,
            "num_points": all_people_data.shape[0]  # número de linhas
        })

    return outliers

def calculateDensityOutliers(num_outliers_per_activity):
    print("------- DENSITY ------")
    for activity in range(NUM_ACTIVITIES):
        d = num_outliers_per_activity[activity]["num_outliers"] / num_outliers_per_activity[activity]["num_points"] * 100
        print(f"{activities.loc[activity, "name"]}: {d}")
    return

def main():
    # EX 2
    getFiles(PATH)                 # get all the individuals
    ind = getIndividual(PATH, 0)    # get one individual
    #print(ind)
    #print(calculateModule(ind[0], 1, 3))

    # EX 3.1
    num_outliers_per_activity = boxPlot_modules(plot = False)   # still with outliers

    # EX 3.2 - analyse outliers
    calculateDensityOutliers(num_outliers_per_activity)

    return

if __name__ == "__main__":
    main()