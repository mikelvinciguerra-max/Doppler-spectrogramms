import gc
import glob
import numpy as np
import scipy.io as sio
from scipy.ndimage import median_filter
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
            print(f"Loading error for {file_name}. Error: {e}")
            continue

        if not {'BW', 'CSI', 'Subcarriers'}.issubset(mat_data.keys()):
            print(f'Variables missing in {file_name}. Ignored.')
            continue

        BW = mat_data['BW'].item()
        CSI = mat_data['CSI']
        Subcarriers = mat_data['Subcarriers'].item()
        
        # Determine A or B variant based on 7th character of filename (index 6)
        variant = file_name[6] if len(file_name) > 6 else 'A'

        if BW == 20:
            indices_to_remove = [] if Subcarriers == 56 else REMOVE_20_DEFAULT
        elif BW == 80:
            indices_to_remove = REMOVE_80B if variant == 'B' else REMOVE_80A
        else:
            print(f'BW not supported ({BW}) in {file_name}')
            continue

        if indices_to_remove:
            CSI = np.delete(CSI, indices_to_remove, axis=1)

        output_path = os.path.join(output_folder, file_name)
        sio.savemat(output_path, {'CSI': CSI})
        print(f'Processed : {file_name}')

    print('Processing completed.')


def linear_phase_transformation(csi_phase):
    """
    Applique une transformation linéaire pour atténuer le décalage temporel
    (offset mismatches/CFO/SFO) comme décrit dans l'article.
    """
    unwrapped_phase = np.unwrap(csi_phase, axis=1)
    K = unwrapped_phase.shape[1]
    
    # Indices centrés : évite le biais -epsilon_s*(K-1)/2 par paquet
    m = np.arange(K) - (K - 1) / 2
    
    # Pente par régression sur TOUTES les sous-porteuses (robuste au bruit
    # ponctuel), au lieu de seulement les 2 extrémités
    epsilon_s = (unwrapped_phase @ m) / (m @ m)
    tau_s = np.mean(unwrapped_phase, axis=1)
    
    epsilon_s = epsilon_s[:, np.newaxis]
    tau_s = tau_s[:, np.newaxis]
    m_matrix = m[np.newaxis, :]
    
    calibrated_phase = unwrapped_phase - (epsilon_s * m_matrix) - tau_s
    return calibrated_phase

def remove_amplitude_outliers(csi_complex, window=5, seuil=3.5):
    """
    Filtre de Hampel : détecte/corrige les paquets aberrants (sauts d'AGC,
    glitches) sur l'amplitude, sous-porteuse par sous-porteuse.
    """
    amplitude = np.abs(csi_complex)
    phase = np.angle(csi_complex)
    
    mediane = median_filter(amplitude, size=(window, 1), mode='nearest')
    mad = median_filter(np.abs(amplitude - mediane), size=(window, 1), mode='nearest') + 1e-9
    
    aberrants = np.abs(amplitude - mediane) > seuil * 1.4826 * mad
    amplitude_corrigee = np.where(aberrants, mediane, amplitude)
    
    touches = np.sum(aberrants, axis=1) > 0.3 * amplitude.shape[1]
    if touches.any():
        print(f"  • {touches.sum()} paquet(s) aberrant(s) corrigé(s) (AGC/glitch probable)")
    
    return amplitude_corrigee * np.exp(1j * phase)

def tsfr_then_complex_to_amplitude_phase(file_name):
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Linear Phase Transform + Amplitude-Phase: {os.path.basename(file_name)}")
    print(f"{'='*60}")
    
    data = sio.loadmat(file_name)
    csi = data['CSI']
    # csi = remove_amplitude_outliers(csi)
    print(f"  • Input shape: {csi.shape}")
    print(f"  • Data type: {csi.dtype}")
    
    # Extraction de l'amplitude et de la phase
    csi_amplitude = np.abs(csi)
    csi_phase = np.angle(csi)
    
    # Remplacement du detrend temporel par la transformation linéaire
    csi_phase_calibrated = linear_phase_transformation(csi_phase)
    
    # Reconstitution du signal complexe avec la phase assainie
    csi_processed = csi_amplitude * np.exp(1j * csi_phase_calibrated)
    
    # Conversion finale en format Amplitude/Phase
    csi_a = np.abs(csi_processed).astype(np.float32)
    csi_p = np.angle(csi_processed).astype(np.float32)
    
    csi_output = np.stack((csi_a, csi_p), axis=-1)
    data['CSI'] = csi_output
    output_path = "./data_preprocessed/"+os.path.basename(file_name)
    sio.savemat(output_path, data)
    
    elapsed = time.time() - start_time
    print(f"  • Output shape: {csi_output.shape}")
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
    
    print(f"  • Computing mean subtraction...")
    
    static_mean = np.mean(csi_matrix, axis=0, keepdims=True)
    filtered_csi = csi_matrix - static_mean
    
    # RAM Cleanup
    del csi_matrix
    del static_mean
    gc.collect()
    
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


if __name__ == "__main__":
    total_start = time.time()
    
    print("\n" + "#"*60)
    print("#  PREPROCESSING PIPELINE START")
    print("#"*60)

    # Phase 1 : Subcarrier Removal
    input_folder = "data_ehunam"
    output_folder = "data_preprocessed"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"\n[Phase 1/3][Subcarrier Removal] Processing files from '{input_folder}'...")
    process_subcarriers(input_folder, output_folder)
    
    # Phase 2: Linear Phase Sanitization + Amplitude-Phase 
    folder = "data_preprocessed"
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    print(f"\n[Phase 2/3][Phase Transform + Amplitude-Phase] Processing {len(files)} files from '{folder}'...")
    for file in files:
        file_path = os.path.join(folder, file)
        tsfr_then_complex_to_amplitude_phase(file_path)

    # Phase 3: Mean Subtraction
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    print(f"\n[Phase 3/3][Mean Subtraction] Processing {len(files)} files from '{folder}'...")
    for file in files:
        file_path = os.path.join(folder, file)
        data = sio.loadmat(file_path)
        process_and_save_csi_mean_subtraction(data, var_name='CSI', file_path=file_path, overwrite=True)
    
    total_elapsed = time.time() - total_start
    print("\n" + "#"*60)
    print(f"#  PREPROCESSING PIPELINE COMPLETED in {total_elapsed:.2f}s")
    print("#"*60 + "\n")