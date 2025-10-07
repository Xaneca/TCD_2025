import numpy as np
import matplotlib as plt
import pandas as pd

PATH = "./../../FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master"
NUM_PEOPLE = 15
NUM_SENSORS = 5
individuals = []    # tam 15 -> tam 5 -> tam 12 + data
                    # (15, 5, 12)

def getFiles(path):
    for i in range(NUM_PEOPLE):
        ind = []
        for j in range(NUM_SENSORS):
            df = pd.read_csv(f'{path}/part{i}/part{i}dev{j + 1}.csv')
            ind.append(df)
        individuals.append(ind)
    return

def getIndividual(path, ind_num):
    ind = []
    for j in range(NUM_SENSORS):
        df = pd.read_csv(f'{path}/part{ind_num}/part{ind_num}dev{j + 1}.csv')
        ind.append(df.to_numpy())

    return ind

def calculateModule(ind, sensor_num, i_x, i_z):
    return np.sqrt(np.sum(ind[sensor_num][:, i_x:i_z]**2, axis=1))   # axis = 1 -> por linha

def boxPlot_modules():
    plt.figure()
    

    return

def main():
    # EX 2
    getFiles(PATH)
    #ind = getIndividual(PATH, 0)
    #print(ind)

    # EX 3


    return

if __name__ == "__main__":
    main()