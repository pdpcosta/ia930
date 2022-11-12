import numpy as np
from os import listdir
from scipy import signal, stats, spatial
from tqdm import tqdm

DATA_DIR = 'data/'
FREQ_RATE = 24 # Frquência de amostragem dos sensores
WINDOW_SIZE = 3 # Tamanho das janelas para extração de features em segundos
OVERLAP = 0.1

# Função de filtragem definida separadamente para controle dos parâmetros
def filter_noise(seq):
    return signal.medfilt(seq, kernel_size=3)

def extract_features(window):
    '''
    Input:
    np.array -> shape = (window_size, sensors_axis)

    Output:
    List[float]

    Recebe uma janela de n amostragem do sinal de acelerometro e
    giroscópio e retorna uma lista com as features calculadas para
    cada eixo dos sensores
    '''

    features = []

    # Feature por eixo do sensor
    # média, mediana, max, min, range, interquartile range, variância
    # desvio padrão, desvio absoluto médio, assimetria, curtose,
    # RMS, energia, potência, taxa de cruzamento de zero, taxa de mudança de
    # sinal da inclinação, entropia

    for sensor_axis in range(window.shape[1]):
        sensor_data = window[:, sensor_axis]

        features.append(np.mean(sensor_data)) # Média

        features.append(np.median(sensor_data)) # Mediana

        max, min = np.max(sensor_data), np.min(sensor_data)

        features.append(max) # Máximo

        features.append(min) # Mínimo

        features.append(max-min) # Range

        features.append(np.percentile(sensor_data, 75) - np.percentile(sensor_data, 25)) # Qautile Range

        features.append(np.var(sensor_data)) # Variância

        features.append(np.std(sensor_data)) # Desvio Padrão

        features.append(np.mean(np.abs(sensor_data - np.mean(sensor_data)))) # Desvio Absoluto Médio

        features.append(stats.skew(sensor_data)) # Assimetria

        features.append(stats.kurtosis(sensor_data)) # Curtose

        features.append(np.sqrt(np.mean(sensor_data**2))) # RMS

        features.append(np.mean(sensor_data**2)) # Energia

        features.append(np.sqrt(np.sum(sensor_data**2)) / sensor_data.shape[0]) # Potência

        features.append(np.nonzero(np.diff(sensor_data > 0))[0].size / sensor_data.shape[0]) # Taxa de Cruzamento de Zero

        ds_dt = sensor_data[1:] - sensor_data[:-1]
        d2s_dt2 = ds_dt[1:] - ds_dt[:-1]
        features.append(np.nonzero(d2s_dt2)[0].size / sensor_data.shape[0]) # Mudança de inclinação

#        norm = sensor_data / max
#        p = norm / np.sum(norm)
#        features.append(stats.entropy(p)) # Entropia


    # Features por sensor (dependem da combinação dos eixos x, y e z)
    # Magnitude, área de magnitude do sinal, ângulos da média dos valores
    # jerk médio, desvio padrão do jerk

    features.append(np.mean(np.sqrt(np.sum(window**2, axis=1)))) # Magnitude do Sinal

    features.append(np.mean(np.sum(np.abs(window), axis=1))) # Área do Sinal Magnitude

    ds_dt = window[1:, :] - window[:-1, :]
    jerk = np.sqrt(np.sum(ds_dt**2, axis=1))
    features.append(np.mean(jerk)) # Jerk Médio
    features.append(np.std(jerk)) # Desvio Padrão do Jerk

    accelerometer_data = window[:, :3]

    mean_acc = np.mean(accelerometer_data, axis=0)
    features.append(spatial.distance.cosine(mean_acc, [1, 0, 0])) # Ângulo com eixo x
    features.append(spatial.distance.cosine(mean_acc, [0, 1, 0])) # Ângulo com eixo y
    features.append(spatial.distance.cosine(mean_acc, [0, 0, 1])) # Ângulo com eixo z

    return features

def load_and_extract(root_dir):
    processed_data = []

    pbar = tqdm(sorted(listdir(root_dir)), total=len(listdir(root_dir)))
    for file_name in pbar:
        # Carrega os dados do arquivo
        data = np.loadtxt(root_dir + file_name, delimiter=',')

        # Extrai coluna contendo os labels de emoção
        labels = data[:,1]

        # Extrai os dados capturados para cada label
        organized_data = []
        organized_data.append(data[np.where(labels == -1)[0], 2:-1])
        organized_data.append(data[np.where(labels == 0)[0], 2:-1])
        organized_data.append(data[np.where(labels == 1)[0], 2:-1])

        # Aplica filtro de média móvel ao longo dos sinais dos sensores
        organized_data = [np.apply_along_axis(filter_noise, 0, segment) for segment in organized_data]

        # Para cada conjunto de dados de sensores, referentes as três emoções,
        # é feita a segmentação em janelas de amostras referentes a WINDOW_SIZE segundos
        segment_size = WINDOW_SIZE * FREQ_RATE
        segmented_data = []
        for segment in organized_data:
            step = int(segment_size * (1 - OVERLAP))
            idxs = np.array([list(range(i, i+segment_size)) for i in range(0, segment.shape[0]-segment_size, step)])
            segmented_data.append(segment[idxs])

        # Converte cada janela de amostragem em um vetor de features junto com seu label
        vectorized_data = []
        for label in range(len(segmented_data)):
            for window in segmented_data[label]:
                vectorized_data.append(extract_features(window[:, :3]) + extract_features(window[:, 3:]) + [label])

        processed_data.extend(vectorized_data)

    return np.asarray(processed_data)

if __name__ == "__main__":
    preprocess_data(DATA_DIR)
