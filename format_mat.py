import os
from scipy.io import loadmat
import numpy as np
import scipy.io as sio

def complex_to_amplitude_phase(file_name):
    print(f"Processing file: {os.path.basename(file_name)}")
    data = loadmat(file_name)
    csi = data['CSI']
    csi_a = np.zeros(csi.shape, dtype=np.float32)
    csi_p = np.zeros(csi.shape, dtype=np.float32)
    for i in range(csi.shape[0]):
        for j in range(csi.shape[1]):
            complex_value = csi[i, j]
            amplitude = abs(complex_value)
            phase = np.angle(complex_value)
            csi_a[i, j] = amplitude
            csi_p[i, j] = phase
    csi = np.stack((csi_a, csi_p), axis=-1)
    data['CSI'] = csi
    sio.savemat("./data_preprocessed/"+os.path.basename(file_name), data)
    print(f"Saved processed file: ./data_preprocessed/{file_name}")

dossier = "data_ehunam"

for file in os.listdir(dossier):
    if file.endswith(".mat"):
        file_path = os.path.join(dossier, file)
        complex_to_amplitude_phase(file_path)