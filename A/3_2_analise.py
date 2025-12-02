import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Heatmap simples
import seaborn as sns
import pandas as pd

# Accelerometer
acc = [4.12, 0.64, 1.89, 6.40, 7.21, 7.40, 7.34, 19.94, 20.05, 17.39, 18.21, 12.07, 6.17, 14.06, 6.52, 6.38]
# Gyroscope
gyr = [8.94, 5.22, 7.74, 2.11, 2.10, 1.57, 1.82, 9.92, 12.49, 11.79, 12.77, 3.02, 9.56, 2.44, 2.68, 2.30]
# Magnetometer
mag = [0.30, 5.26, 5.66, 2.19, 2.23, 2.94, 3.05, 4.99, 6.51, 5.68, 9.58, 3.67, 3.79, 5.23, 5.56, 5.71]


# densidades já calculadas: dict[v][atividade] = densidade
densidades = {
    'ACC': acc,
    'GYR': gyr,
    'MAG': mag
}

activities = pd.read_csv("nameActivities.csv")
print(activities)

# Média e desvio padrão por módulo
for v in densidades:
    media = np.mean(densidades[v])
    desvio = np.std(densidades[v])
    print(f"{v}: média={media:.2f}%, desvio={desvio:.2f}%")

df = pd.DataFrame(densidades, index=activities['id'])
sns.heatmap(df, annot=True, fmt=".2f", cmap="YlOrRd")
plt.show()
