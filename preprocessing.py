import gc
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import detrend
import os
import time

REMOVE_20_DEFAULT = [0, 1, 2, 3, 32, 61, 62, 63]
REMOVE_80A = [*range(6), *range(127, 131), *range(251, 256)]
REMOVE_80B = [*range(6), 32, *range(59, 70), 96, *range(123, 134), 160, *range(186, 198), 224, *range(251, 256)]

def process_subcarriers(folder, output_folder):
    mat_files = glob.glob(os.path.join(folder, '*.mat'))
    if not mat_files:
        print('No .mat files found.')
        return

    for file_path in mat_files:
        file_name = os.path.basename(file_path)
        
        try:
            mat_data = sio.loadmat(file_path)
        except Exception as e:
            print(f"Erreur de chargement pour {file_name}. (Utilisez h5py pour v7.3). Erreur: {e}")
            continue

        if not {'BW', 'CSI', 'Subcarriers'}.issubset(mat_data.keys()):
            print(f'Variables manquantes dans {file_name}. Ignoré.')
            continue

        BW = mat_data['BW'].item()
        CSI = mat_data['CSI']
        Subcarriers = mat_data['Subcarriers'].item()
        
        env_data = mat_data.get('Enviroment', '')
        env_str = str(env_data[0]) if isinstance(env_data, np.ndarray) and env_data.size else str(env_data)

        if BW == 20:
            indices_to_remove = [] if Subcarriers == 56 else REMOVE_20_DEFAULT
        elif BW == 80:
            indices_to_remove = REMOVE_80B if 'Industrial Laboratory' in env_str else REMOVE_80A
        else:
            print(f'BW non supporté ({BW}) dans {file_name}')
            continue

        if indices_to_remove:
            CSI = np.delete(CSI, indices_to_remove, axis=1)

        output_path = os.path.join(output_folder, file_name)
        sio.savemat(output_path, {'CSI': CSI})
        print(f'Traité : {file_name}')

    print('Processing completed.')

def complex_to_amplitude_phase(file_name):
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Processing file: {os.path.basename(file_name)}")
    print(f"{'='*60}")
    
    data = sio.loadmat(file_name)
    csi = data['CSI']
    print(f"  • Input shape: {csi.shape}")
    print(f"  • Data type: {csi.dtype}")
    
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
    output_path = "./data_preprocessed/"+os.path.basename(file_name)
    sio.savemat(output_path, data)
    
    elapsed = time.time() - start_time
    print(f"  • Output shape: {csi.shape}")
    print(f"✓ Saved processed file: {output_path}")
    print(f"  • Time elapsed: {elapsed:.2f}s")


def process_and_save_csi_mean_subtraction(data, var_name, file_path, overwrite=False):
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Mean Subtraction Filtering: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    if var_name not in data:
        raise KeyError(f"Variable '{var_name}' not found in the .mat file.")
    
    csi_matrix = data[var_name]
    print(f"  • Loaded CSI matrix dimensions: {csi_matrix.shape}")
    print(f"  • Memory usage: {csi_matrix.nbytes / 1024**2:.2f} MB")
    
    time_axis = np.argmax(csi_matrix.shape)
    if time_axis != 0:
        print(f"  • Auto-transposing (time axis was {time_axis})...")
        csi_matrix = np.swapaxes(csi_matrix, 0, time_axis)
    
    # Mean Subtraction (Static Clutter Removal)
    print(f"  • Computing mean subtraction...")
    
    # Calculate the mean along the time axis (axis 0)
    # keepdims=True ensures the shape stays compatible for vectorized subtraction
    static_mean = np.mean(csi_matrix, axis=0, keepdims=True)
    
    filtered_csi = csi_matrix - static_mean
    
    # RAM Cleanup
    del csi_matrix
    del static_mean
    gc.collect()
    
    # Revert transposition if axes were swapped
    if time_axis != 0:
        filtered_csi = np.swapaxes(filtered_csi, 0, time_axis)
    
    if overwrite:
        data[var_name] = filtered_csi
        print(f"  • Variable '{var_name}' successfully overwritten.")
    else:
        new_var_name = f"{var_name}_filtered"
        data[new_var_name] = filtered_csi
        print(f"  • Filtered data appended as '{new_var_name}'.")
    
    sio.savemat(file_path, data)
    elapsed = time.time() - start_time
    print(f"✓ Processing completed in {elapsed:.2f}s")

# def load_and_sanitize_mat(file_path):
#     mat_data = sio.loadmat(file_path)
#     csi_raw = mat_data['CSI'] 
    
#     if csi_raw.shape[0] < 400:
#         return None
    
#     return csi_raw

def tsfr_preprocessing(data, file_path):
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"TSFR Preprocessing: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    csi_data = data['CSI']
    print(f"  • Processing CSI data shape: {csi_data.shape}")
    
    for i in range(csi_data.shape[0]):
        for j in range(csi_data.shape[1]):
            complex_value = csi_data[i, j]
            amplitude = abs(complex_value)
            phase = np.angle(complex_value)
            phase_corrected = np.unwrap(phase, discont=np.pi, axis=0)
            phase_detrended = detrend(phase, axis=0)
            csi_data[i, j] = amplitude * np.exp(1j * phase)
    
    sio.savemat(file_path, data)
    elapsed = time.time() - start_time
    print(f"✓ TSFR preprocessing completed in {elapsed:.2f}s")
    

if __name__ == "__main__":
    total_start = time.time()
    
    print("\n" + "#"*60)
    print("#  PREPROCESSING PIPELINE START")
    print("#"*60)

    # Pahse 1 : Subcarrier Removal
    input_folder = "data_ehunam"
    output_folder = "data_preprocessed"
    print(f"\n[Phase 1/4][Subcarrier Removal] Processing {len(os.listdir(input_folder))} files from '{input_folder}'...")
    process_subcarriers(input_folder, output_folder)
    
    # Phase 2: Complex to Amplitude-Phase
    folder = "data_preprocessed"
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    print(f"\n[Phase 2/4][Complex to Amplitude-Phase] Processing {len(files)} files from '{folder}'...")
    for file in files:
        file_path = os.path.join(folder, file)
        complex_to_amplitude_phase(file_path)

    # Phase 3 & 4: Mean Subtraction + TSFR
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    print(f"\n[Phase 3/4][Mean Subtraction] Processing {len(files)} files from '{folder}'...")
    for file in files:
        file_path = os.path.join(folder, file)
        data = sio.loadmat(file_path)
        process_and_save_csi_mean_subtraction(data, var_name='CSI', file_path=file_path, overwrite=False)
        tsfr_preprocessing(data, file_path)
    
    total_elapsed = time.time() - total_start
    print("\n" + "#"*60)
    print(f"#  PREPROCESSING PIPELINE COMPLETED in {total_elapsed:.2f}s")
    print("#"*60 + "\n")

    
