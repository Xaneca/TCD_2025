from PIL import Image
import os

# Caminho da pasta com as imagens
folder_path = r"./plots/ex3_1" 
# folder_path = r"./plots/ex3_4" 

# Lista todos os arquivos de imagem da pasta
im_paths = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Organiza as imagens por tipo (ACC, GYR, MAG)
# 3_1
acc_images = sorted([f for f in im_paths if 'vector0' in f])
gyr_images = sorted([f for f in im_paths if 'vector1' in f])
mag_images = sorted([f for f in im_paths if 'vector2' in f])

# 3_4
# acc_images = sorted([f for f in im_paths if 'Accelerometer' in f])
# gyr_images = sorted([f for f in im_paths if 'Gyroscope' in f])
# mag_images = sorted([f for f in im_paths if 'Magnetometer' in f])

# Agora queremos **colunas = variáveis**, linhas = sensores
columns = [acc_images, gyr_images, mag_images]

# Número de linhas e colunas
cols = len(columns)                 # 3 variáveis
rows = max(len(col) for col in columns)  # número de sensores

# Pega tamanho da primeira imagem como referência
w, h = Image.open(os.path.join(folder_path, acc_images[0])).size

# Cria imagem vazia para a grade
new_im = Image.new('RGB', (cols * w, rows * h))

# Preenche a grade
for col_index, col in enumerate(columns):
    for row_index, filename in enumerate(col):
        img = Image.open(os.path.join(folder_path, filename))
        img = img.resize((w, h))  # garante mesmo tamanho
        x = col_index * w         # coluna → eixo x
        y = row_index * h         # sensor → eixo y
        new_im.paste(img, (x, y))

# Salva e mostra
new_im.save(os.path.join(folder_path, "grid_sensores_colunas.png"))
new_im.show()
