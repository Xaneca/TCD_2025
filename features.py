import numpy as np
from scipy.stats import skew, kurtosis, iqr, pearsonr
from numpy.fft import fft, fftfreq

# Parâmetros da Janela
SAMPLING_RATE = 100  # Hz [6]
WINDOW_LENGTH = 200  # 2 segundos * 100 Hz [5]
GRAVITY_G_UNITS = 1.0  # Aproximando a gravidade como 1g (assumindo que os dados de aceleração são em unidades de 'g')

################################################################################
# A. ROTINAS DE CARACTERÍSTICAS ESTATÍSTICAS (TEMPORAIS E ESPECTRAIS)
################################################################################

def extract_mean(data):
    """Média (Mean)"""
    return np.mean(data)

def extract_median(data):
    """Mediana (Median)"""
    return np.median(data)

def extract_std_dev(data):
    """Desvio Padrão (Standard Deviation)"""
    return np.std(data)

def extract_variance(data):
    """Variância (Variance)"""
    return np.var(data)

def extract_rms(data):
    """Root Mean Square (RMS)"""
    return np.sqrt(np.mean(data**2))

def extract_avg_derivatives(data):
    """Derivadas Médias (Averaged derivatives)"""
    # Usando a diferença finita para aproximação da derivada
    derivative = np.diff(data) * SAMPLING_RATE
    return np.mean(derivative)

def extract_skewness(data):
    """Assimetria (Skewness)"""
    # Nota: SciPy calcula a skewness.
    return skew(data)

def extract_kurtosis(data):
    """Curtose (Kurtosis)"""
    # Nota: SciPy calcula a curtose.
    return kurtosis(data)

def extract_iqr(data):
    """Interquartile Range (IQR)"""
    return iqr(data)

def extract_zcr(data):
    """Zero Crossing Rate (ZCR)"""
    # Número de vezes que o sinal muda de positivo para negativo ou vice-versa
    # Exclui o primeiro ponto, garante que o cálculo é feito em relação à mudança.
    zero_crossings = np.nonzero(np.diff(np.sign(data)))
    return len(zero_crossings) / WINDOW_LENGTH

def extract_mcr(data):
    """Mean Crossing Rate (MCR)"""
    mean_value = np.mean(data)
    data_centered = data - mean_value
    mean_crossings = np.nonzero(np.diff(np.sign(data_centered)))
    return len(mean_crossings) / WINDOW_LENGTH

def extract_spectral_entropy(data):
    """Entropia Espectral (Spectral Entropy)"""
    # Calcular a magnitude do espectro ).
    N = len(data)
    fft_magnitudes = np.abs(fft(data))

    # Excluímos a primeira componente (DC) e consideramos apenas a primeira metade (espectro de potência unilateral)
    power_spectrum = fft_magnitudes[1:N//2]**2

    # Normalizar o espectro para obter a distribuição de probabilidade
    if np.sum(power_spectrum) == 0:
        return 0

    P = power_spectrum / np.sum(power_spectrum)

    # Calcular a entropia espectral (H = - sum(P * log2(P)))
    # Evitar log(0)
    P = P[P > 0]

    # Usando log base 2 para entropia
    spectral_entropy = -np.sum(P * np.log2(P))
    return spectral_entropy

# Funções auxiliares para características físicas espectrais (DF e ENERGY)
def _calculate_fft_power(data):
    """Calcula a magnitude quadrada do FFT (Espectro de Potência)."""
    N = len(data)
    # FFT dos dados
    yf = fft(data)
    # Magnitude quadrada (potência)
    power_spectrum = np.abs(yf[:N//2])**2
    # Frequências correspondentes
    xf = fftfreq(N, 1/SAMPLING_RATE)[:N//2]
    return xf, power_spectrum

def extract_dominant_frequency(data):
    """Frequência Dominante (DF)"""
    xf, power_spectrum = _calculate_fft_power(data)

    # Excluímos 0Hz/DC [16]
    if len(power_spectrum) <= 1:
        return 0.0

    # Encontrar o índice da máxima potência (excluindo DC, que é o primeiro índice)
    max_index = np.argmax(power_spectrum[1:]) + 1
    return xf[max_index]

def extract_energy(data):
    """Energia (ENERGY)"""
    N = len(data)
    _, power_spectrum = _calculate_fft_power(data)

    # Somamos os componentes, excluindo o primeiro (DC) 
    total_energy = np.sum(power_spectrum[1:])

    # Normalizado pelo comprimento da janela
    return total_energy / N

################################################################################
# B. ROTINAS DE CARACTERÍSTICAS FÍSICAS (TEMPORAIS E ESPECTRAIS DERIVADAS)
################################################################################

# Pressuposto de orientação do sensor: x=Gravidade, y/z=Heading

def _remove_gravity(Ax, Ay, Az):
    """Calcula a aceleração vetorial após remover a gravidade estática."""

    # Calcular o vetor de aceleração Média (que representa a gravidade em repouso)
    avg_A = np.array([np.mean(Ax), np.mean(Ay), np.mean(Az)])

    # Subtrair a média. Isso é uma aproximação comum para remover a gravidade.
    Ax_lin = Ax - avg_A[0]
    Ay_lin = Ay - avg_A[1]
    Az_lin = Az - avg_A[2]

    return Ax_lin, Ay_lin, Az_lin

def extract_ai_vi(Ax, Ay, Az):
    """AI (Mean Intensity) e VI (Variance Intensity)"""

    # Calcular aceleração sem gravidade (Ax_lin, Ay_lin, Az_lin)
    Ax_lin, Ay_lin, Az_lin = _remove_gravity(Ax, Ay, Az)

    # Calcular a Intensidade de Movimento (MI)
    MI = np.sqrt(Ax_lin**2 + Ay_lin**2 + Az_lin**2)

    # AI (Averaged Intensity)
    AI = np.mean(MI)

    # VI (Variance Intensity)
    VI = np.var(MI)

    return AI, VI

def extract_sma(Ax, Ay, Az):
    """Normalized Signal Magnitude Area (SMA)"""
    # Nota: Esta definição usa as acelerações brutas (ax, ay, az).
    N = len(Ax)

    # Soma das magnitudes absolutas dos três eixos
    SMA = (np.sum(np.abs(Ax)) + np.sum(np.abs(Ay)) + np.sum(np.abs(Az))) / N
    return SMA

def extract_eva(Ax, Ay, Az):
    """Eigenvalues of Dominant Directions (EVA)"""

    # Montar a matriz de dados [T x 3]
    A_data = np.vstack([Ax, Ay, Az]).T

    # Calcular a matriz de covariância
    Cov_matrix = np.cov(A_data, rowvar=False) # rowvar=False indica que as variáveis são colunas

    #Calcular os autovalores
    eigenvalues = np.linalg.eigvalsh(Cov_matrix)

    # Ordenar em ordem decrescente (Os maiores correspondem às direções dominantes)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    # Usar os dois primeiros autovalores
    # O maior é associado à direção da gravidade (vertical) e o segundo, à direção de progressão 
    return eigenvalues_sorted[0], eigenvalues_sorted[1]

def extract_cagh(Ax, Ay, Az):
    """Correlation between Acceleration along Gravity and Heading Directions (CAGH)"""

    # Direção da gravidade é o eixo x
    A_gravity = Ax

    # Norma Euclidiana da aceleração na direção de progressão (heading
    A_heading_norm = np.sqrt(Ay**2 + Az**2)

    # Correlação entre as duas direções
    # pearsonr retorna (coeficiente, p-value), queremos apenas o coeficiente
    correlation, _ = pearsonr(A_gravity, A_heading_norm)
    return correlation

def cumtrapz_numpy(y, dx):
        y = np.asarray(y)
        y_avg = (y[:-1] + y[1:]) / 2
        dx_arr = np.full_like(y_avg, dx)
        cum_int = np.concatenate(([0], np.cumsum(y_avg * dx_arr)))
        return cum_int

def extract_avh(Ax, Ay, Az, SAMPLING_RATE):
    """Averaged Velocity along Heading Direction (AVH)"""
    dt = 1.0 / SAMPLING_RATE

    # Remover gravidade (assumindo que tens a função _remove_gravity definida)
    Ax_lin, Ay_lin, Az_lin = _remove_gravity(Ax, Ay, Az)
    

    Vy_inst = cumtrapz_numpy(Ay_lin, dx=dt)
    Vz_inst = cumtrapz_numpy(Az_lin, dx=dt)

    # Velocidades médias ao longo da janela
    Avg_Vy = np.mean(Vy_inst)
    Avg_Vz = np.mean(Vz_inst)

    # Norma Euclidiana dessas velocidades médias
    AVH = np.sqrt(Avg_Vy**2 + Avg_Vz**2)
    return AVH

def extract_avg(Ax, dt):
    """Averaged Velocity along Gravity Direction (AVG)"""

    # Aceleração linear ao longo da gravidade
    Ax_lin = Ax - np.mean(Ax)  # Aproximação para remover a gravidade estática

    # Velocidade instantânea através da integração numérica
    Vx_inst = cumtrapz_numpy(Ax_lin, dx=dt)

    # Média da velocidade instantânea
    AVG = np.mean(Vx_inst)
    return AVG

def extract_aratg(Gx, dt):
    """Averaged Rotation Angles related to Gravity Direction (ARATG)"""

    # Assumimos que Gx é em dps (graus por segundo)
    Angle_x_inst = cumtrapz_numpy(Gx, dx=dt)

    # Rotação cumulativa
    Cumulative_Angle = Angle_x_inst[-1]

    # Normalizado pelo comprimento da janela
    ARATG = Cumulative_Angle / WINDOW_LENGTH

    return ARATG

def extract_aae_are(Ax, Ay, Az, Gx, Gy, Gz):
    """AAE e ARE (Averaged Energy): Média da Energia espectral dos eixos do acelerómetro e giroscópio [16, 17]."""

    # Energia para cada eixo (usando a rotina já definida)
    E_Ax = extract_energy(Ax)
    E_Ay = extract_energy(Ay)
    E_Az = extract_energy(Az)
    E_Gx = extract_energy(Gx)
    E_Gy = extract_energy(Gy)
    E_Gz = extract_energy(Gz)

    # AAE: Média da Energia dos três eixos de aceleração
    AAE = np.mean([E_Ax, E_Ay, E_Az])

    # ARE: Média da Energia dos três eixos do giroscópio
    ARE = np.mean([E_Gx, E_Gy, E_Gz])

    return AAE, ARE

def extract_pairwise_correlation(data1, data2):
    """Pairwise Correlation"""
    # pearsonr retorna (coeficiente, p-value)
    correlation, _ = pearsonr(data1, data2)
    return correlation

def extract_feature_vector(window_data):

    Ax = window_data[:, 1]
    Ay = window_data[:, 2]
    Az = window_data[:, 3]
    Gx = window_data[:, 4]
    Gy = window_data[:, 5]
    Gz = window_data[:, 6]

    # Parâmetros
    ACC_CHANNELS = [Ax, Ay, Az]
    GYRO_CHANNELS = [Gx, Gy, Gz]
    ALL_CHANNELS = ACC_CHANNELS + GYRO_CHANNELS

    dt = 1.0 / SAMPLING_RATE
    feature_vector = []

    # 1. CARACTERÍSTICAS ESTATÍSTICAS (Aplicadas a cada um dos 6 canais)

    # Características temporais
    for channel in ALL_CHANNELS:
        feature_vector.extend([
            extract_mean(channel),              # Mean 
            extract_median(channel),            # Median
            extract_std_dev(channel),           # Standard Deviation 
            extract_variance(channel),          # Variance
            extract_rms(channel),               # RMS
            extract_avg_derivatives(channel),   # Averaged derivatives
            extract_skewness(channel),          # Skewness
            extract_kurtosis(channel),          # Kurtosis
            extract_iqr(channel),               # IQR 
            extract_zcr(channel),               # ZCR 
            extract_mcr(channel)                # MCR
        ])

    # Características espectrais (DF, ENERGY, Entropy)
    for channel in ALL_CHANNELS:
        feature_vector.extend([
            extract_dominant_frequency(channel), # DF 
            extract_energy(channel),             # ENERGY
            extract_spectral_entropy(channel)    # Spectral Entropy
        ])

    # Correlação Par a Par (Pairwise Correlation)
    # Total: C(6, 2) = 15 pares.

    # Acelerómetro
    feature_vector.append(extract_pairwise_correlation(Ax, Ay))
    feature_vector.append(extract_pairwise_correlation(Ax, Az))
    feature_vector.append(extract_pairwise_correlation(Ay, Az))

    # Giroscópio
    feature_vector.append(extract_pairwise_correlation(Gx, Gy))
    feature_vector.append(extract_pairwise_correlation(Gx, Gz))
    feature_vector.append(extract_pairwise_correlation(Gy, Gz))

    feature_vector.append(extract_pairwise_correlation(Ax, Gx))
    feature_vector.append(extract_pairwise_correlation(Ay, Gy))
    feature_vector.append(extract_pairwise_correlation(Az, Gz))
    feature_vector.append(extract_pairwise_correlation(Ax, Gy))
    feature_vector.append(extract_pairwise_correlation(Ax, Gz))
    feature_vector.append(extract_pairwise_correlation(Ay, Gx))
    feature_vector.append(extract_pairwise_correlation(Ay, Gz))
    feature_vector.append(extract_pairwise_correlation(Az, Gx))
    feature_vector.append(extract_pairwise_correlation(Az, Gy))


    # 2. CARACTERÍSTICAS FÍSICAS (DERIVADAS/FUSÃO)

    # 1. AI e VI (Movement Intensity)
    AI, VI = extract_ai_vi(Ax, Ay, Az)
    feature_vector.extend([AI, VI])

    # 2. SMA (Signal Magnitude Area)
    feature_vector.append(extract_sma(Ax, Ay, Az))

    # 3. EVA (Eigenvalues)
    EVA1, EVA2 = extract_eva(Ax, Ay, Az)
    feature_vector.extend([EVA1, EVA2])

    # 4. CAGH (Correlation Gravity/Heading)
    feature_vector.append(extract_cagh(Ax, Ay, Az))

    # 5. AVH (Averaged Velocity Heading)
    feature_vector.append(extract_avh(Ax, Ay, Az, SAMPLING_RATE))

    # 6. AVG (Averaged Velocity Gravity)
    feature_vector.append(extract_avg(Ax, dt))

    # 7. ARATG (Averaged Rotation Angles Gravity)
    feature_vector.append(extract_aratg(Gx, dt))

    # 10 & 11. AAE e ARE (Averaged Energy)
    AAE, ARE = extract_aae_are(Ax, Ay, Az, Gx, Gy, Gz)
    feature_vector.extend([AAE, ARE])

    return np.array(feature_vector)
